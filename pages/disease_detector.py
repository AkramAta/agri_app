import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import json
import os
import requests

# ✅ Download from Hugging Face
def download_model_from_url(url, destination):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)

# 📍 Model location and URL
model_path = "models/plant_disease/plant_disease_model.h5"
model_url = "https://huggingface.co/datasets/Akram-11/plant-disease-model/resolve/main/plant_disease_model.h5"

# ⬇️ Download if missing
if not os.path.exists(model_path):
    st.info("Downloading model from Hugging Face... ⏳")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model_from_url(model_url, model_path)

# ✅ Load model
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
