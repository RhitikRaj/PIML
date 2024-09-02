import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Load the trained model
model_path = '/workspaces/PIML/Final_model.h5/Final_model.h5'
model = tf.saved_model.load(model_path)

# Define image size
IMAGE_SHAPE = (224, 224)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image, dtype=np.float32)  # Ensure float32 data type
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Tool Wear Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict the class
    # Convert to TensorFlow tensor with explicit float32 data type
    tf_image = tf.convert_to_tensor(processed_image, dtype=tf.float32)
    
    # Get the concrete function for prediction
    infer = model.signatures['serving_default']
    prediction = infer(tf_image)
    
    # Extract the output tensor (adjust the key if necessary)
    output_key = list(prediction.keys())[0]
    predicted_class = tf.argmax(prediction[output_key], axis=-1)
    
    class_names = ['New', 'Medium', 'Defective']  # Example class names
    st.write(f"Prediction: {class_names[predicted_class[0]]}")

# Print model information for debugging
print("Model signatures:", list(model.signatures.keys()))
print("Serving default inputs:", model.signatures['serving_default'].inputs)
print("Serving default outputs:", model.signatures['serving_default'].outputs)