import streamlit as st
import pandas as pd
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder


# MODEL AUDIT LOG
def log_action(message):
    os.makedirs("logs", exist_ok=True)
    with open("logs/model_audit_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")


def retraining_page():

    st.title("Model Retraining & Evaluation")

    st.write("Upload a dataset to retrain a new model and compare it with the existing deployed model.")

    uploaded_file = st.file_uploader("Upload Dataset (CSV file)", type=["csv"])

    if uploaded_file is not None:

        log_action("Dataset uploaded in retraining module")

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        if target:

            X = df.drop(columns=[target])
            y = df[target]

            log_action(f"Target selected: {target}")

            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == "object":
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])

            # Target handling
            if pd.api.types.is_numeric_dtype(y):

                if y.nunique() > 10:
                    st.warning("⚠️ Continuous numeric target detected. Converting into classes...")
                    y = pd.qcut(y, q=3, labels=[0, 1, 2])

            else:
                st.info("Categorical target detected. Encoding labels automatically...")
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # OUTLIER DETECTION 
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso.fit(X_train_scaled)

            outlier_preds = iso.predict(X_test_scaled)
            num_outliers = sum(outlier_preds == -1)

            st.subheader("🧪 Outlier Detection")
            st.write(f"Detected {num_outliers} potential outlier(s) in test data")

            log_action(f"Outlier detection run | {num_outliers} outliers detected")

            if num_outliers > 0:
                st.warning("⚠️ Some patients are unusual compared to training data")

            # Show outlier results
            outlier_df = X_test.copy()
            outlier_df["Outlier"] = outlier_preds

            st.write("Outlier Detection Results (1 = Normal, -1 = Outlier)")
            st.dataframe(outlier_df.head())

            # Show only true outliers
            outliers_only = outlier_df[outlier_df["Outlier"] == -1]

            if not outliers_only.empty:
                st.error("🚨 Outlier Patients Detected:")
                st.dataframe(outliers_only)

            model_choice = st.selectbox(
                "Choose Model",
                ["Random Forest", "Logistic Regression"]
            )

            if st.button("Train New Model"):

                log_action(f"Training started using {model_choice}")

                if model_choice == "Random Forest":
                    new_model = RandomForestClassifier()
                else:
                    new_model = LogisticRegression(max_iter=1000)

                new_model.fit(X_train_scaled, y_train)

                preds = new_model.predict(X_test_scaled)

                new_acc = accuracy_score(y_test, preds)
                new_f1 = f1_score(y_test, preds, average="weighted")
                new_prec = precision_score(y_test, preds, average="weighted")
                new_rec = recall_score(y_test, preds, average="weighted")

                log_action(f"Model trained | Accuracy={new_acc:.4f} | F1={new_f1:.4f}")

                st.subheader("🆕 New Model Performance")
                st.write(f"Accuracy: {new_acc:.3f}")
                st.write(f"F1 Score: {new_f1:.3f}")
                st.write(f"Precision: {new_prec:.3f}")
                st.write(f"Recall: {new_rec:.3f}")

                st.subheader("Detailed Classification Report")
                st.text(classification_report(y_test, preds))

                # EXISTING MODEL
                try:
                    st.subheader("Existing Model Performance")

                    with open("models/model_report.txt", "r") as f:
                        report_text = f.read()

                    def extract_metric(text, key):
                        for line in text.split("\n"):
                            if key in line:
                                try:
                                    return float(line.split(":")[1].strip())
                                except:
                                    return None
                        return None

                    old_acc = extract_metric(report_text, "Accuracy")
                    old_f1 = extract_metric(report_text, "F1 Score")
                    old_prec = extract_metric(report_text, "Precision")
                    old_rec = extract_metric(report_text, "Recall")

                    if old_acc is not None:

                        st.subheader("Model Comparison")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🆕 New Model")
                            st.write(f"Accuracy: {new_acc:.3f}")
                            st.write(f"F1 Score: {new_f1:.3f}")
                            st.write(f"Precision: {new_prec:.3f}")
                            st.write(f"Recall: {new_rec:.3f}")

                        with col2:
                            st.markdown("### 📦 Existing Model")
                            st.write(f"Accuracy: {old_acc}")
                            st.write(f"F1 Score: {old_f1}")
                            st.write(f"Precision: {old_prec}")
                            st.write(f"Recall: {old_rec}")

                        if new_acc > old_acc:
                            st.success("✅ New model performs better than existing model")
                            log_action("New model outperformed existing model")
                        else:
                            st.warning("⚠️ Existing model still performs better")
                            log_action("Existing model performed better than new model")

                    else:
                        st.warning("Could not extract metrics from model_report.txt")

                except FileNotFoundError:
                    st.warning("⚠️ model_report.txt not found in models folder")