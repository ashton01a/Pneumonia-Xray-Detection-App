import tensorflow as tf

# Load your original model
model = tf.keras.models.load_model('pneumonia_detector.h5')

# Set up converter for post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
compressed_model = converter.convert()

# Save the compressed model
with open('compressed_pneumonia_model.tflite', 'wb') as f:
    f.write(compressed_model)

print("✅ Compressed model saved as compressed_pneumonia_model.tflite")
