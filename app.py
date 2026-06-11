import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("rice_disease_v1_fixed.keras")

class_names = ['Bacterial Blight', 'Brown Spot', 'Healthy', 'Tungro']

st.title("🌾 Rice Disease Detector")

file = st.file_uploader("Upload rice leaf image")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    st.write("Prediction:", class_names[np.argmax(pred)])
    st.write("Confidence:", f"{float(np.max(pred))*100:.2f}%")
