import streamlit as st  
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import datetime

from dashboard import main
from dataset_upload import dataset_page
from database import create_table, insert_patient, insert_audit, get_audit_logs
from records import records_page
from model_retraining import retraining_page 


st.set_page_config(page_title="HealthPredict", layout="wide")


# LOGIN SYSTEM
users = {
    "doctor": "health123",
    "admin": "predict456"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login page
if not st.session_state.logged_in:

    st.markdown("""
<div class="login-background"></div>
<div class="login-container">
    <div class="login-title">HealthPredict</div>
    <div class="login-subtitle">Secure Diabetes Risk System</div>
</div>
""", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username   # IMPORTANT
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


# Load CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# CREATE DATABASE TABLES
create_table()


# AUDIT LOG PAGE
def audit_page():

    st.title("🔐 Audit Log System")

    st.write("All predictions are logged for transparency and accountability.")

    try:
        df = get_audit_logs()

        if df.empty:
            st.info("No audit logs found yet.")
            return

        st.subheader("System Activity Log")
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("High Risk Cases", len(df[df["risk"] == "High"]))
        col3.metric("Moderate Cases", len(df[df["risk"] == "Moderate"]))

        st.download_button(
            "⬇ Download Audit Log",
            df.to_csv(index=False),
            "audit_log.csv",
            "text/csv"
        )

        st.subheader("🔎 Filter by Risk Level")

        risk_filter = st.selectbox("Select risk level", ["All", "Low", "Moderate", "High"])

        if risk_filter != "All":
            st.dataframe(df[df["risk"] == risk_filter], use_container_width=True)

    except Exception as e:
        st.error(f"Error loading audit logs: {e}")


# Prediction Page Function
def prediction_page():

    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pima_df = pd.read_csv("data/diabetes.csv")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.title("HealthPredict")

    st.sidebar.header("Patient Information")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    age = st.sidebar.number_input("Age", 1, 120, 30)
    preg = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
    bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
    skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Risk"):

        prob = model.predict_proba(input_scaled)[0][1]
        risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.7 else "High"

        st.subheader(f"Predicted Risk: {risk} ({prob*100:.1f}%)")

        # SESSION HISTORY
        st.session_state.history.append({
            "Age": age,
            "Pregnancies": preg,
            "Glucose": glucose,
            "BP": bp,
            "Skin": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Risk": risk,
            "Probability": prob
        })

        # DATABASE SAVE
        insert_patient((
            age, preg, glucose, bp, skin, insulin, bmi, dpf, risk, prob
        ))

        # AUDIT LOG SAVE
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = st.session_state.get("username", "unknown")

        insert_audit((
            username,
            age,
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            risk,
            prob,
            timestamp
        ))

        # SHAP
        st.subheader("Feature Impact (SHAP) - Current Prediction")

        features = ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"]

        try:
            background = scaler.transform(pima_df.iloc[:100, :-1])
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(input_scaled)

            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1][0]
            else:
                shap_values_to_plot = shap_values[0]

            shap_values_to_plot = np.array(shap_values_to_plot).flatten()

            if len(shap_values_to_plot) != len(features):
                shap_values_to_plot = shap_values_to_plot[:len(features)]

            df_shap = pd.DataFrame({
                "Feature": features,
                "Impact": shap_values_to_plot
            }).sort_values(by="Impact", key=abs, ascending=False)

            plt.figure()
            plt.barh(df_shap["Feature"], df_shap["Impact"])
            plt.xlabel("SHAP Impact")
            plt.title("Feature Impact for This Patient")
            plt.gca().invert_yaxis()
            st.pyplot(plt)
            plt.clf()

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")

    # HISTORY
    if st.session_state.history:

        st.subheader("Prediction History")
        df_history = pd.DataFrame(st.session_state.history)

        st.dataframe(df_history)

        st.download_button(
            "Download CSV",
            df_history.to_csv(index=False),
            "history.csv"
        )

    # MODEL VISUALS
    st.subheader("Model Evaluation Visuals")

    for file, caption in [
        ("models/confusion_matrix.png","Confusion Matrix"),
        ("models/roc_curve.png","ROC Curve"),
        ("models/feature_importance.png","Feature Importance")
    ]:
        if os.path.exists(file):
            st.image(file, caption=caption)

    st.subheader("Limitations")
    st.write("""
    - Dataset only contains female Pima Indians.
    - Small dataset, not clinically validated.
    - Educational use only.
    """)

    st.subheader("About HealthPredict")
    st.write("""
    Machine Learning system for diabetes risk prediction with explainability and logging.
    """)


# NAVIGATION MENU
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Prediction", "Dashboard", "Dataset Upload", "Records", "Audit Log", "Model Retraining"]  
)

if page == "Prediction":
    prediction_page()
elif page == "Dashboard":
    main()
elif page == "Dataset Upload":
    dataset_page()
elif page == "Records":
    records_page()
elif page == "Audit Log":
    audit_page()
elif page == "Model Retraining":   
    retraining_page()