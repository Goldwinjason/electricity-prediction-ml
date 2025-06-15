import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Sample Data - Replace with your dataset
# Example CSV columns: ['hour', 'temperature', 'humidity', 'previous_usage', 'consumption']
data = pd.read_csv("electricity_data.csv")

# Features and label
X = data[['hour', 'temperature', 'humidity', 'previous_usage']]
y = data['consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, "electricity_model.pkl")
