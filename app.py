import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Set Streamlit page config
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩻",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="compressed_pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# App Title
st.title('🩻 Pneumonia Detection from Chest X-Ray')
st.subheader('Upload a chest X-ray image below to get a prediction!')

# Upload file section
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)

    # Resize and normalize
    img_resized = cv2.resize(image, (150, 150))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    probability = prediction * 100

    # Show prediction
    if prediction > 0.5:
        st.error(f"🛑 Pneumonia detected with {probability:.2f}% confidence!")
    else:
        st.success(f"✅ Normal X-ray detected with {(100 - probability):.2f}% confidence!")
else:
    st.info('👈 Please upload a chest X-ray image to get started.')