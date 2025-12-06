import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import requests

# -----------------------------
# ✅ Function to download model
# -----------------------------
def download_model_from_url(url, destination):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)

# -----------------------------
# 📍 Model location and URL
# -----------------------------
model_path = "models/plant_disease/mobilenetAICropPrediction.keras"
model_url = "https://huggingface.co/datasets/Akram-11/plant-disease-model/resolve/main/mobilenetAICropPrediction.keras"

# -----------------------------
# ⬇️ Download model if missing
# -----------------------------
if not os.path.exists(model_path):
    st.info("Downloading model from Hugging Face... ⏳")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model_from_url(model_url, model_path)
    st.success("✅ Model downloaded successfully!")

# -----------------------------
# ✅ Load model
# -----------------------------
model = tf.keras.models.load_model(model_path, compile=False)

# -----------------------------
# ✅ Load class index mapping
# -----------------------------
class_indices_path = "models/plant_disease/class_indices.json"
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# -----------------------------
# 🌿 Streamlit UI
# -----------------------------
st.header("🦠 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted_class = list(class_indices.keys())[class_idx]

    # Display result
    st.success(f"🦠 Detected disease: **{predicted_class}** with confidence: **{confidence:.2f}%**")
