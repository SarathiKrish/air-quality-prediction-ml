import streamlit as st
import pandas as pd
from skops import io as sio

model = sio.load("model/air_quality_model.skops")

st.title("🌫️ Air Quality Prediction App")
st.markdown("Predict AQI levels using machine learning.")

# Input features
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
so2 = st.number_input("SO2")
no2 = st.number_input("NO2")
co = st.number_input("CO")
o3 = st.number_input("O3")

if st.button("Predict AQI"):
    input_data = [[pm25, pm10, so2, no2, co, o3]]
    prediction = model.predict(input_data)
    st.success(f"Predicted AQI Level: {prediction[0]}")
