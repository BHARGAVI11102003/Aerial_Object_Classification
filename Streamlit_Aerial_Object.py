import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("bird_drone_model.keras")

st.title("Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).resize((224,224))

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = prediction[0][0]

    if confidence > 0.5:
        st.write(f"Prediction: Drone")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.write(f"Prediction: Bird")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")