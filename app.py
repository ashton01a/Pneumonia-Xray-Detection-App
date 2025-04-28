import streamlit as st
import numpy as np
import cv2

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩻",
    layout="centered",
    initial_sidebar_state="auto",
)

# App title and instructions
st.title('🩻 Pneumonia Detection from Chest X-Ray')
st.subheader('Upload a chest X-ray image below to get started!')

# Upload file section
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
else:
    st.info('👈 Please upload an X-ray image to begin.')