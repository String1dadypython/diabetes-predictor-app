import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    user_data = np.array([[pregnancies, glucose, blood_pressure,
                           skin_thickness, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Likely to have diabetes.")
    else:
        st.success("✅ Unlikely to have diabetes.")
