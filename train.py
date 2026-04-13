import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# AUDIT LOG 
def log_action(message):
    os.makedirs("logs", exist_ok=True)
    with open("logs/model_audit_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")


# Start pipeline log
log_action("Training pipeline started")

# Create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# Load dataset
data = pd.read_csv("data/diabetes.csv")
log_action("Dataset loaded: data/diabetes.csv")

# Replace invalid zeros with median
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero_invalid:
    data[col] = data[col].replace(0, np.nan)
    data[col].fillna(data[col].median(), inplace=True)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    "KNN": {'n_neighbors': [3, 5, 7, 9]}
}

best_model = None
best_model_name = ""
best_accuracy = 0

# Train + tune models
for name, model in models.items():

    log_action(f"Training model: {name}")

    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=5)
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    log_action(f"{name} accuracy: {acc:.4f}")

    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_model = model
        best_accuracy = acc
        best_model_name = name

log_action(f"Best model selected: {best_model_name} | Accuracy={best_accuracy:.4f}")

print(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save model and scaler
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

log_action("Model and scaler saved to disk")

# FINAL EVALUATION
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_proba, y_test)
roc_auc = auc(fpr, tpr)

log_action(f"Final evaluation completed | Accuracy={accuracy:.4f} | F1={f1:.4f}")

# SAVE VISUALS
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{best_model_name} Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{best_model_name} ROC Curve")
plt.legend()
plt.savefig("models/roc_curve.png")
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_

    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=X.columns)
    plt.title(f"{best_model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()

# SAVE REPORT
with open("models/model_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

log_action("Model report and visualisations saved")

print("Model, metrics, and visualizations saved successfully.")