# Importing required libraries
import numpy as np
import pickle  # To load the model
import streamlit as st
import pandas as pd
import requests
import os

# LOADING THE MODEL FROM GITHUB (RAW LINK)
url = "https://raw.githubusercontent.com/ClintonJuma/Machine-Learning-model/main/heartdisease_model.sav"

# Define model file path
model_filename = "heartdisease_model.sav"

# Download the file if it doesn't exist
if not os.path.exists(model_filename):
    response = requests.get(url)
    with open(model_filename, "wb") as f:
        f.write(response.content)

# Load the saved model
with open(model_filename, "rb") as f:
    loaded_model = pickle.load(f)

# Prediction function
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    return "The Person Has Heart Disease" if prediction[0] == 1 else "The Person Does Not Have Heart Disease"

# Streamlit UI
def main():
    st.title("Heart Disease Prediction Machine Learning Model")

    # Collecting user input
    age = st.number_input("Age (15-80)", min_value=15, max_value=80, step=1)
    sex = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    Chest_Pain = st.selectbox("Chest Pain Level (0-3)", [0, 1, 2, 3])
    Blood_Pressure = st.number_input("Blood Pressure (94-200 mmHg)", min_value=94, max_value=200, step=1)
    cholestoral = st.number_input("Cholesterol Level (131-290 mg/dl)", min_value=131, max_value=290, step=1)
    Fasting_Blood_Sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_electrocardiographic = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    Maximum_Heart_Rate = st.number_input("Maximum Heart Rate (99-162)", min_value=99, max_value=162, step=1)
    Excersize_Includes = st.radio("Exercise Induced Angina", [0, 1])
    ST_Depression = st.number_input("ST Depression (0.0 - 4.4)", min_value=0.0, max_value=4.4, step=0.1)
    Slope_of_Excersize = st.selectbox("Slope of Exercise (0-2)", [0, 1, 2])
    Number_of_vessels = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    Thalassemia = st.selectbox("Thalassemia (1-3)", [1, 2, 3])

    # Prediction
    diagnosis = ""
    if st.button("PREDICT"):
        diagnosis = heart_disease_prediction([
            age, sex, Chest_Pain, Blood_Pressure, cholestoral, Fasting_Blood_Sugar,
            resting_electrocardiographic, Maximum_Heart_Rate, Excersize_Includes,
            ST_Depression, Slope_of_Excersize, Number_of_vessels, Thalassemia
        ])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
