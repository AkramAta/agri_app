import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import json
import os
import requests


def download_model_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)


model_path = "models/plant_disease/plant_disease_model.h5"
file_id = "1BHlkeVUp2ieawY4X8-hPqBPC6xK2srV1"  # 👈 Replace this with your actual file ID


if not os.path.exists(model_path):
    st.info("Downloading model... Please wait ⏳")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model_from_drive(file_id, model_path)


model = load_model(model_path, compile=False)


with open("models/plant_disease/class_indices.json", "r") as f:
    class_indices = json.load(f)

# UI
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
