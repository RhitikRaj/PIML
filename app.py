import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import os


# Load the trained model
# model_path = '/workspaces/PIML/Finalmodel'
# model = tf.saved_model.load(model_path) 



# Assuming 'Finalmodel' is in the same directory as app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'Finalmodel_SavedModel')
model = tf.saved_model.load(model_path) 
# model=tf.keras.models.load_model(model_path)
IMAGE_SHAPE = (224, 224)

# Function to preprocess the image (consistent with training preprocessing)
def preprocess_image(image):
    image = image.resize(IMAGE_SHAPE)  # Resize to expected input size
    image = np.array(image, dtype=np.float32)  # Convert to a numpy array
    # image = image / 255.0  # Normalize pixel values to [0, 1]
    
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
    st.write(prediction[output_key])
    
    # Example class names (ensure they match the order used during model training)
    # class_names = ['New', 'Medium', 'Defective']
    # Ensure this order matches the training phase
    class_names = ['Defective', 'Medium', 'New']

    
    # # Display the predicted class name
    st.write(f"Prediction: {class_names[predicted_class[0]]}")
    # 7. Perform prediction
    # prediction = model.predict(processed_image)

    # 8. Display raw prediction for debugging
    # st.write("Raw prediction output:", prediction)

    # 9. Get predicted class index
    # predicted_class = np.argmax(prediction, axis=-1)

    # 10. Get confidence score
    # confidence = np.max(prediction, axis=-1)

    # 11. Display predicted class index, name, and confidence
    # st.write(f"Predicted class index: {predicted_class[0]}")
    # st.write(f"Predicted class name: {class_names[predicted_class[0]]}")
    # st.write(f"Confidence: {confidence[0]*100:.2f}%")



# import tensorflow as tf
# import streamlit as st
# from PIL import Image
# import numpy as np
# import os
# from tensorflow import keras

# # Get current working directory
# current_dir = os.getcwd()
# model_path = os.path.join(current_dir, 'Finalmodel_SavedModel')

# # Wrap the SavedModel in a TFSMLayer
# tfsm_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# # Create a Keras model that uses the TFSMLayer
# inputs = keras.Input(shape=(224, 224, 3), dtype='float32')
# outputs = tfsm_layer(inputs)
# model = keras.Model(inputs, outputs)

# IMAGE_SHAPE = (224, 224)

# # Function to preprocess the image (consistent with training preprocessing)
# def preprocess_image(image):
#     image = image.convert('RGB')  # Ensure image is in RGB format
#     image = image.resize(IMAGE_SHAPE)  # Resize to expected input size
#     image = np.array(image, dtype=np.float32)  # Convert to a numpy array
#     image = image / 255.0  # Normalize pixel values to [0, 1]
#     return image

# # Streamlit UI
# st.title("Tool Wear Classification")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     # Preprocess the image
#     processed_image = preprocess_image(image)
#     processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

#     # Perform prediction
#     prediction = model.predict(processed_image)

#     # Display raw prediction for debugging
#     st.write("Raw prediction output:", prediction)

#     # Get predicted class index
#     predicted_class = np.argmax(prediction, axis=-1)

#     # Get confidence score
#     confidence = np.max(prediction, axis=-1)

#     # Define class names
#     class_names = ['Defective', 'Medium', 'New']  # Ensure this matches your model's classes

#     # Display predicted class index, name, and confidence
#     # st.write(f"Predicted class index: {predicted_class[0]}")
#     # st.write(f"Predicted class name: {class_names[predicted_class[0]]}")
#     st.write(f"Confidence: {confidence[0]*100:.2f}%")
