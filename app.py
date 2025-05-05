import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🌫️ Air Quality Prediction App")
st.markdown("Predict air quality using manually entered values.")

# Manual input
pm25 = st.number_input("Enter PM2.5 (µg/m³)", 0.0)
pm10 = st.number_input("Enter PM10 (µg/m³)", 0.0)
so2 = st.number_input("Enter SO₂ (µg/m³)", 0.0)
no2 = st.number_input("Enter NO₂ (µg/m³)", 0.0)

if st.button("Predict AQI Level"):
    features = [[pm25, pm10, so2, no2]]
    prediction = model.predict(features)
    st.success(f"Predicted AQI Level: **{prediction[0]}**")
