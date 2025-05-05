import streamlit as st
import pandas as pd
import joblib

# Title
st.title("🌫️ Air Quality Prediction App")
st.subheader("Naan Mudhalvan Project - B.E. CSE")

# Load model
model = joblib.load("model.pkl")

# Input fields
st.markdown("### Enter Pollution Levels")
pm25 = st.number_input("PM2.5 (µg/m³)", 0.0)
pm10 = st.number_input("PM10 (µg/m³)", 0.0)
so2 = st.number_input("SO₂ (µg/m³)", 0.0)
no2 = st.number_input("NO₂ (µg/m³)", 0.0)

if st.button("Predict Air Quality Index (AQI)"):
    features = [[pm25, pm10, so2, no2]]
    prediction = model.predict(features)
    st.success(f"Predicted AQI Level: {prediction[0]}")
