import streamlit as st
import pandas as pd
from database import get_all_patients

def records_page():
    st.title("Patient Records Database")

    data = get_all_patients()

    if data:
        df = pd.DataFrame(data, columns=[
            "ID","Age","Pregnancies","Glucose","BP",
            "Skin","Insulin","BMI","DPF","Risk","Probability"
        ])

        st.dataframe(df)

        search = st.text_input("Search by Risk")

        if search:
            filtered = df[df["Risk"].str.contains(search, case=False)]
            st.dataframe(filtered)
    else:
        st.info("No patient records found.")