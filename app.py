import streamlit as st
import pandas as pd
from skops import io as sio
import numpy as np

# Page title
st.set_page_config(page_title="Air Quality Prediction", layout="centered")

st.title("🌿 Air Quality Level Predictor")
st.markdown("Upload data or input values to predict the **Air Quality Level** using a trained machine learning model.")

# Load the model (skops)
@st.cache_resource
def load_model():
    model = sio.load("model/air_quality_model.skops")
    return model

model = load_model()

# Example input fields (customize based on your dataset columns)
def user_input_features():
    col1 = st.number_input("CO2 Level (ppm)", min_value=0.0, value=400.0)
    col2 = st.number_input("NO2 Level (ppb)", min_value=0.0, value=20.0)
    col3 = st.number_input("O3 Level (ppb)", min_value=0.0, value=30.0)
    col4 = st.number_input("PM2.5 Level (µg/m3)", min_value=0.0, value=12.0)
    col5 = st.number_input("Temperature (°C)", value=25.0)
    
    data = {
        'CO2': col1,
        'NO2': col2,
        'O3': col3,
        'PM2.5': col4,
        'Temperature': col5
    }
    
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Predict and display
if st.button("Predict Air Quality Level"):
    prediction = model.predict(input_df)
    label = int(prediction[0])  # assuming it's encoded

    # You can map this integer to actual air quality labels if needed
    label_map = {
        0: "Good",
        1: "Moderate",
        2: "Unhealthy for Sensitive Groups",
        3: "Unhealthy",
        4: "Very Unhealthy",
        5: "Hazardous"
    }

    result = label_map.get(label, f"Level {label}")
    st.success(f"Predicted Air Quality Level: **{result}**")
