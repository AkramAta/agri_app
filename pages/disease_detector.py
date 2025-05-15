import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import json

# Load the model
model = load_model("models/plant_disease/plant_disease_model.h5", compile=False)

# Load class indices
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

    # Map the predicted class index to the corresponding class name
    predicted_class = list(class_indices.keys())[class_idx]

    st.success(f"🦠 Detected disease: **{predicted_class}** with confidence: **{confidence:.2f}%**")
