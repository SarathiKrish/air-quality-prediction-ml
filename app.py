import streamlit as st
import pandas as pd
from skops import io as sio

# Page configuration
st.set_page_config(page_title="Air Quality Prediction", layout="centered")

st.title("🌍 Air Quality Level Predictor")
st.markdown("Enter pollution values to predict the **Air Quality Index Level (AQI)**.")

# Load model using skops
@st.cache_resource
def load_model():
    return sio.load("model/air_quality_model.skops")

model = load_model()

# Input section for the required features
def user_input_features():
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=35.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=70.0)
    so2 = st.number_input("SO2 (ppb)", min_value=0.0, value=10.0)
    no2 = st.number_input("NO2 (ppb)", min_value=0.0, value=20.0)

    data = {
        'PM2.5': pm25,
        'PM10': pm10,
        'SO2': so2,
        'NO2': no2
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Predict button
if st.button("Predict AQI Level"):
    prediction = model.predict(input_df)
    level = int(prediction[0])

    # You can define this mapping based on your dataset’s original labels
    aqi_map = {
        0: "Good",
        1: "Moderate",
        2: "Unhealthy for Sensitive Groups",
        3: "Unhealthy",
        4: "Very Unhealthy",
        5: "Hazardous"
    }

    result = aqi_map.get(level, f"Level {level}")
    st.success(f"Predicted AQI Level: **{result}**")
