import streamlit as st
import pandas as pd
import pickle as pkl
import joblib

with open("models/pump/pump.pkl", "rb") as file:
    model = pkl.load(file)



scaler = joblib.load("models/pump/scaler.pkl")


st.header("💧 Pump Control System")

temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0)
humidity = st.number_input("💨 Air Humidity (%)", min_value=0.0, max_value=100.0)
soil_moisture = st.number_input("🌱 Soil Moisture", min_value=0.0, max_value=1000.0)

if st.button("🔍 Check Pump Status"):
    input_data = pd.DataFrame([[soil_moisture, temperature, humidity]],
                              columns=['Soil Moisture', 'Temperature', 'Air Humidity'])


    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ The pump should be turned ON.")
    else:
        st.warning("🛑 The pump should remain OFF.")
