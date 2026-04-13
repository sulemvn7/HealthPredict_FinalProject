import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("📊 HealthPredict - Pima Dataset Analysis Dashboard")

    # Load the Pima dataset
    pima_df = pd.read_csv("data/diabetes.csv")

    st.subheader("Dataset Overview")
    st.dataframe(pima_df.head())
    st.write(f"Total records: {len(pima_df)}")

    # Histogram of Age
    fig_age = px.histogram(
        pima_df, x="Age", nbins=20,
        title="Age Distribution",
        color_discrete_sequence=["#00c9ff"]
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Histogram of Pregnancies
    fig_preg = px.histogram(
        pima_df, x="Pregnancies", nbins=15,
        title="Pregnancies Distribution",
        color_discrete_sequence=["#2ECC71"]
    )
    st.plotly_chart(fig_preg, use_container_width=True)

    # Histogram of Glucose Levels
    fig_glucose = px.histogram(
        pima_df, x="Glucose", nbins=20,
        title="Glucose Distribution",
        color_discrete_sequence=["#F1C40F"]
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

    # Correlation heatmap
    feature_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    corr = pima_df[feature_cols + ["Outcome"]].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',
        zmin=-1, zmax=1,
        hoverongaps=False
    ))
    fig_heatmap.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Boxplot of BMI by Outcome
    fig_bmi = px.box(
        pima_df, x="Outcome", y="BMI", color="Outcome",
        title="BMI Distribution by Diabetes Outcome",
        color_discrete_map={0: "#2ECC71", 1: "#E74C3C"}
    )
    st.plotly_chart(fig_bmi, use_container_width=True)

    # Boxplot of Glucose by Outcome
    fig_glu_outcome = px.box(
        pima_df, x="Outcome", y="Glucose", color="Outcome",
        title="Glucose Levels by Diabetes Outcome",
        color_discrete_map={0: "#2ECC71", 1: "#E74C3C"}
    )
    st.plotly_chart(fig_glu_outcome, use_container_width=True)