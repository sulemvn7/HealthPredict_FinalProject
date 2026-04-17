# HealthPredict - Machine Learning Diabetes Prediction System

## Overview

HealthPredict is a machine learning web application that predicts diabetes using patient data and allows retraining, evaluation, and monitoring of models.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Core Features

* 🔮 Disease prediction using trained ML models
* 🧠 Model retraining with custom datasets
* 📊 Model evaluation (Accuracy, F1 Score, Precision, Recall)
* 🔄 Model comparison (new vs existing model)
* 🧾 Model audit logging system (tracks uploads, training, results)
* 🧪 Outlier detection system (detects abnormal patients)
* 📋 Classification reports for detailed analysis
* 💾 Model deployment (replace existing model)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚙️ Setup & Run Instructions

1. Clone the repository link from link.txt.

2. Install dependencies

Make sure Python (3.10+) is installed, then run:

pip install -r requirements.txt

3.Run this in the terminal to create the models folder:

python train.py

4. Run the application

streamlit run app.py

5. Open in browser

Go to:

http://localhost:8501

6.Login Details(You can use either set of details to log in):

Username:doctor, Password:health123

or 

Username:admin, Password:predict456

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Dependencies

* streamlit
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib
* shap
* plotly

Install all using:


pip install -r requirements.txt

## 🔐 Configuration / Environment

Ensure the following folders exist:

models/

logs/

data/

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Sample Usage / Test Inputs

* Use the provided datasets in the test folder to test the dataset upload and model retraining features.:

* Example features:

  * Glucose
  * BloodPressure
  * BMI
  * Insulin
  * Age

Upload either dataset in the Model Retraining or Dataset Upload section to test functionality.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Known Limitations

* Works best with structured/tabular datasets only
* Categorical encoding is basic (Label Encoding)
* No real-time API deployment yet
* Model versioning is not implemented (overwrites current model)
* Outlier detection is dataset-based (not per individual prediction yet)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Summary

This project demonstrates a complete machine learning pipeline including:

* Data preprocessing
* Model training and evaluation
* Model comparison and deployment
* Logging and monitoring
* Anomaly detection

It simulates a real-world healthcare system.
