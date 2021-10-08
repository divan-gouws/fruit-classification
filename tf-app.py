import streamlit as st
import tensorflow as tf
from tensorflow import keras
from Pillow import Image, ImageOps
import numpy as np
import pickle

with open("labels.txt", "rb") as fp:
    labels = pickle.load(fp)

def teachable_machine_classification(img, weights_file):
    model = tf.keras.models.load_model('fruit_94-97.hdf5')
    data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.float32)
    image = img
    size = (100, 100)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    nomralized_image_array = (image_array.astype(np.float32) / 255)
    data[0] = nomralized_image_array
    prediction_percentage = model.predict(data)
    prediction = prediction_percentage.round()
    return prediction, prediction_percentage

st.title("Fruit classification")
st.sidebar.title("Parameters")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
    label = None
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    label, perc = teachable_machine_classification(image, 'fruit_94-97.hdf5')
    label_index = np.argmax(label, axis=None, out=None)
    st.write("The uploaded fruit is a(n) {} with {:%} confidence.".format(labels[label_index], perc[0, label_index]))