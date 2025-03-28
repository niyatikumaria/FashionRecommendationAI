import streamlit as st
import gdown
import numpy as np
import pickle
import os
import tensorflow as tf
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.layers import GlobalMaxPooling2D
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

# Google Drive file IDs
EMBEDDINGS_FILE_ID = "10j-pbgXgIQKw03FsEZsO2B0uP-OT3jom"  # Replace with actual ID
FILENAMES_FILE_ID = "1-8-eY2-7d9j9ylbB4m2pIcVmPVq6AN9n"  # Replace with actual ID

# Download embeddings.pkl
if not os.path.exists("embeddings.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDINGS_FILE_ID}", "embeddings.pkl", quiet=False)

# Download filenames.pkl
if not os.path.exists("filenames.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={FILENAMES_FILE_ID}", "filenames.pkl", quiet=False)

# Load feature list and filenames
with open("embeddings.pkl", "rb") as f:
    feature_list = pickle.load(f)

with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title("Fashion Recommender System")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption="Uploaded Image", use_container_width=True)

    features = feature_extraction(file_path, model)
    indices = recommend(features, feature_list)
    
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(indices[0]):
            col.image(filenames[indices[0][i]], use_container_width=True)
