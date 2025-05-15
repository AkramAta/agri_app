import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import json
import os
import requests

# ✅ Robust Google Drive downloader with token support
def download_model_from_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# 📍 Model location and Drive ID
model_path = "models/plant_disease/plant_disease_model.h5"
file_id = "1BHlkeVUp2ieawY4X8-hPqBPC6xK2srV1"  # 👈 Replace with your own if needed

# ⬇️ Download if missing and validate
if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive... Please wait ⏳")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model_from_drive(file_id, model_path)

    # ❗ Validate file size to detect failure
    if os.path.getsize(model_path) < 100_000:  # <100 KB = likely HTML error page
        st.error("❌ Model download failed. Please check the Google Drive link or quota.")
        st.stop()

# ✅ Load the model
model = load_model(model_path, compile=False)

# ✅ Load class index mapping
with open("models/plant_disease/class_indices.json", "r") as f:
    class_indices = json.load(f)

# 🌿 Streamlit UI
st.header("🦠 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)

    class_idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted_class = list(class_indices.keys())[class_idx]

    st.success(f"🦠 Detected disease: **{predicted_class}** with confidence: **{confidence:.2f}%**")
