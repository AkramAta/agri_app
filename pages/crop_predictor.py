import streamlit as st
import pickle
import numpy as np

# Load the crop recommendation model
with open("models/crop_recommendation/crop_recommendation.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("models/crop_recommendation/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Crop Recommender", layout="centered")

st.title("🌾 Smart Crop Recommendation System")

st.markdown("Provide the environmental and soil details below to get a recommended crop.")

# Updated example inputs
examples = {
    
    "User Example 1": [42, 79, 85, 17.223852, 15.820693, 6.129534, 76.575810, 36.636172],
    "User Example 2": [43, 80, 85, 17.223852, 15.0, 7.129534, 76.0, 35.636172],
    "User Example 3": [36, 76, 75, 18.381204, 16.638052, 8.736338, 70.520567, 39.482547],
    "User Example 4": [35, 78, 75, 18.5, 16.0, 8.0, 71.0, 40.0],
    "User Example 5": [87, 33, 43, 23.0, 83.0, 5.853932, 235.0, 43.679463],
    "User Example 6": [88, 35, 40, 23.579436, 83.587603, 5.853932, 234.030832, 42.679463],
    "User Example 7": [29, 138, 197, 22.190554, 92.437642, 5.830892, 121.662276, 39.181221]
}

# Dropdown for example selection
selected_example = st.selectbox("Select an example input", ["Custom input"] + list(examples.keys()))

if selected_example != "Custom input":
    nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, soil_moisture = examples[selected_example]
else:
    nitrogen = st.number_input("Ratio of Nitrogen (N)", min_value=0.0, step=0.1)
    phosphorous = st.number_input("Ratio of Phosphorous (P)", min_value=0.0, step=0.1)
    potassium = st.number_input("Ratio of Potassium (K)", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, step=0.1)

    if soil_moisture < 30.0 or soil_moisture > 45.0:
        st.warning("⚠️ Soil Moisture value is outside the typical optimal range (30–45%).")

# Prediction trigger
if st.button("Recommend Crop"):
    features = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, soil_moisture]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")
