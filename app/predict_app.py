import os
import numpy as np
import json
from PIL import Image
import streamlit as st
import tensorflow as tf


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f'{working_dir}/TrainedModels/potato_blight_detection_model2.h5'

# Load the pretrained model
model = tf.keras.models.load_model(model_path)

# Loading the class names
class_indices = json.load(open(f'{working_dir}/class_indices.json'))


def preprocess_image(image, target_size):
    # Load the image
    img = Image.open(image)
    # Resize the Image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array 

def predict_image_class(model, image, class_indices):
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]

    return predicted_class_name

# Streamlit App
st.title('Potato Blight Detection')

uploaded_image = st.file_uploader('upload an image...', type=['jpg','jpeg','png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

