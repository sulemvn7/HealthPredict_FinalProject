import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score

def dataset_page():

    st.title("📂 Dataset Upload & Multi-Dataset Testing")

    st.write("""
    Upload any dataset to explore and optionally run predictions.
    """)

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=None  
    )

    if uploaded_file is not None:

        try:
            file_type = uploaded_file.name.split(".")[-1].lower()

            # Read file according to type
            if file_type in ["csv", "txt"]:
                df = pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_type == "json":
                df = pd.read_json(uploaded_file)
            else:
                st.warning(f"File type '{file_type}' not specifically supported. Attempting CSV read.")
                df = pd.read_csv(uploaded_file)

            # Display basic info
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.write("Dataset Shape:", df.shape)

            st.subheader("Automatic Data Exploration")
            st.write("Dataset Statistics")
            st.write(df.describe())
            st.write("Column Types")
            st.write(df.dtypes)

            # Automatic visualisation for numeric columns
            st.subheader("Automatic Data Visualisation")
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            for col in numeric_cols[:5]:
                st.write(f"Distribution of {col}")
                st.bar_chart(df[col])

            # Attempt predictions if compatible columns exist
            try:
                model = joblib.load("models/best_model.pkl")
                scaler = joblib.load("models/scaler.pkl")

                model_features = [
                    "Pregnancies","Glucose","BloodPressure",
                    "SkinThickness","Insulin","BMI",
                    "DiabetesPedigreeFunction","Age"
                ]

                # Only use columns present in the uploaded dataset
                valid_features = [col for col in model_features if col in df.columns]

                if valid_features:
                    X = df[valid_features]
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    df["Prediction"] = predictions
                    st.subheader("Predictions (only using available features)")
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "dataset_predictions.csv"
                    )
                else:
                    st.info("No compatible columns for prediction found. Only dataset exploration is shown.")

            except Exception as e:
                st.error(f"Error running predictions: {e}")

        except Exception as e:
            st.error(f"Unable to read uploaded file: {e}")