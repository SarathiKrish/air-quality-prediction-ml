# 🌿 Air Quality Level Prediction using Machine Learning

This project predicts the **Air Quality Index (AQI) level** based on key air pollutant concentrations using a machine learning model. It is built using `scikit-learn`, deployed with `Streamlit`, and packaged using the modern and version-independent `skops` format.

---

## 📊 Dataset

The dataset includes the following features:

- `PM2.5` – Fine particulate matter (µg/m³)
- `PM10` – Coarse particulate matter (µg/m³)
- `SO2` – Sulfur Dioxide (ppb)
- `NO2` – Nitrogen Dioxide (ppb)

The target label is:

- `AQI_Level` – A categorical variable indicating the air quality level.

---

## 🚀 Features

- Clean UI for real-time AQI level prediction.
- Uses a trained **Random Forest Classifier**.
- Input fields for air pollution data.
- Intelligent label mapping for interpretability (e.g., "Good", "Unhealthy", etc.).
- Fully deployable on **Streamlit Cloud**.

---

## 🧠 Model

The machine learning model was:

- Trained in **Google Colab** using `scikit-learn`.
- Saved in `.skops` format using the `skops` library for **version independence**.
- Loaded efficiently within Streamlit using `@st.cache_resource`.

---

## 🛠 Installation

To run this project locally:

```bash
git clone https://github.com/SarathiKrish/air-quality-prediction-ml.git
cd air-quality-prediction-ml
pip install -r requirements.txt
streamlit run app.py
