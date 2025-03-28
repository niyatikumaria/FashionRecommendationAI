import streamlit as st
import os
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.layers import GlobalMaxPooling2D
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames with proper error handling
try:
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    
    if isinstance(feature_list, dict):
        feature_list = np.array(list(feature_list.values()))
    elif isinstance(feature_list, list):
        feature_list = np.array(feature_list)
    
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
except Exception as e:
    st.error(f"Error loading embedding files: {e}")
    st.stop()

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# Ensure the uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return None

# Upload and process the image
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)
        
        # Extract features
        features = feature_extraction(file_path, model)
        if features is not None:
            indices = recommend(features, feature_list)
            if indices is not None:
                # Display recommended images
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(indices[0]):
                        col.image(filenames[indices[0][i]], use_container_width=True)
    else:
        st.error("File upload failed. Please try again.")
