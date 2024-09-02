import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Final_model.h5')

# Define image size
IMAGE_SHAPE = (224, 224)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image)
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
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=-1)

    class_names = ['New', 'Medium', 'Defective']  # Example class names
    st.write(f"Prediction: {class_names[predicted_class[0]]}")
