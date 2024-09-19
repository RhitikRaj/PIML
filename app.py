import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np


# Load the trained model
model_path = '/workspaces/PIML/Final_model.h5/Final_model.h5'
model = tf.saved_model.load(model_path) 


IMAGE_SHAPE = (224, 224)

# Function to preprocess the image (consistent with training preprocessing)
def preprocess_image(image):
    image = image.resize(IMAGE_SHAPE)  # Resize to expected input size
    image = np.array(image, dtype=np.float32)  # Convert to a numpy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    
    # If image has an alpha channel (RGBA), remove it
    if image.shape[-1] == 4:
        image = image[..., :3]  # Keep only RGB channels
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return image

# Streamlit UI
st.title("Tool Wear Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get the concrete function for prediction from the SavedModel
    infer = model.signatures['serving_default']  # Use the correct signature for inference
    
    # Perform prediction
    prediction = infer(tf.convert_to_tensor(processed_image))
    
    # Extract the output tensor (adjust the key if necessary)
    output_key = list(prediction.keys())[0]
    predicted_class = tf.argmax(prediction[output_key], axis=-1)
    
    # Example class names (ensure they match the order used during model training)
    # class_names = ['New', 'Medium', 'Defective']
    # Ensure this order matches the training phase
    class_names = ['Defective', 'Medium', 'New']

    
    # Display the predicted class name
    st.write(f"Prediction: {class_names[predicted_class[0]]}")