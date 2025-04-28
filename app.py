import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
#model = tf.keras.models.load_model('pneumonia_detector.h5')

# App title
st.title('🩻 Pneumonia Detection from Chest X-Ray')

# Upload file section
uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)

    # Preprocess the image
    img_resized = cv2.resize(image, (150, 150))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_input)[0][0]
    probability = prediction * 100

    # Display the prediction
    if prediction > 0.5:
        st.error(f"Prediction: Pneumonia detected! Confidence: {probability:.2f}%")
    else:
        st.success(f"Prediction: Normal. Confidence: {100 - probability:.2f}%")