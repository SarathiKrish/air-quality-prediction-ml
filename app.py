import streamlit as st
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Check if model exists, else train it
if not os.path.exists("model.pkl"):
    st.warning("🔁 Training model on first run...")

    # Load sample data
    df = pd.read_csv("data/air_quality_data.csv")
    X = df[['PM2.5', 'PM10', 'SO2', 'NO2']]
    y = df['AQI_Level']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    st.success("✅ Model trained and saved.")
else:
    model = joblib.load("model.pkl")

# Streamlit UI
st.title("🌫️ Air Quality Prediction App")
st.write("Enter pollutant levels to predict AQI category.")

pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
so2 = st.number_input("SO₂", min_value=0.0)
no2 = st.number_input("NO₂", min_value=0.0)

if st.button("Predict AQI Level"):
    input_data = [[pm25, pm10, so2, no2]]
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted AQI Level: **{prediction[0]}**")
