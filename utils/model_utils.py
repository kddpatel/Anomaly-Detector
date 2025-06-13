import tensorflow as tf
import numpy as np
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model('model/keras_model.h5')
with open('model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class, confidence
