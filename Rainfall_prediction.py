import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = pd.read_excel('C:\\Users\\ASUS\\Downloads\\New folder\\rainfall_prediction_data.xlsx')

data['Rainfall'] = data['Rainfall'].apply(lambda x: 1 if x == 'Yes' else 0)

features = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Pressure (hPa)', 'Cloud Cover (%)', 'Precipitation (mm)']
X = data[features]
y = data['Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

def predict_rainfall(temperature, humidity, wind_speed, pressure, cloud_cover, precipitation):
    input_data = pd.DataFrame([[temperature, humidity, wind_speed, pressure, cloud_cover, precipitation]], 
                              columns=features)
    prediction = model.predict(input_data)
    return 'Yes' if prediction[0] == 1 else 'No'

example_prediction = predict_rainfall(20, 75, 15, 1008, 50, 2)
print(f"\nPredicted rainfall for the given conditions: {example_prediction}")
