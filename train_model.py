# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the dataset
df = pd.read_csv('data/air_quality_dataset.csv')  # ensure this file exists

# 2. Preprocessing (you can enhance this later)
df = df.dropna()  # remove missing values

# Optional: print column names to identify target and features
print("Columns:", df.columns)

# 3. Select features and target
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]  # example features
y = df['AQI']  # example target (adjust to match your dataset)

# 4. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 7. Save the model
joblib.dump(model, 'model/air_quality_model.pkl')
print("Model saved to model/air_quality_model.pkl")
