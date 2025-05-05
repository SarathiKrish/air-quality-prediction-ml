import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load sample dataset
data = pd.read_csv("data/air_quality_data.csv")

# Features and target
X = data[['PM2.5', 'PM10', 'SO2', 'NO2']]
y = data['AQI_Level']  # Class labels: Good, Moderate, Poor, etc.

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
