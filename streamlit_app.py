# code for keras_model.h5.....................

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Set Streamlit title
# st.title("Teachable Machine Image Classifier")

# # Try loading the Keras model
# try:
#     model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
#     # Optionally convert to .keras format (once)
#     # model.save("model/converted_model.keras", save_format="keras")
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# # Load class labels
# try:
#     with open("model/labels.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]
# except FileNotFoundError:
#     class_names = []
#     st.warning("labels.txt not found. Classes will be shown as index numbers.")

# # Image prediction function
# def predict_image(image: Image.Image):
#     image = image.resize((224, 224))
#     img_array = np.asarray(image).astype(np.float32)
#     if img_array.shape[-1] == 4:  # If image has alpha channel
#         img_array = img_array[..., :3]
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     prediction = model.predict(img_array)[0]
#     top_idx = np.argmax(prediction)
#     label = class_names[top_idx] if class_names else f"Class {top_idx}"
#     confidence = float(prediction[top_idx])
#     return label, confidence

# # File uploader
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     label, confidence = predict_image(image)

#     st.subheader("Prediction:")
#     st.write(f"**Label:** {label}")
#     st.write(f"**Confidence:** {confidence:.2%}")






# code for model.tflite.....................

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Anomaly Detector : Tiles Image Classifier")

# Load labels
try:
    with open("model/labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = []
    st.warning("labels.txt not found. Predictions will show class index.")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image: Image.Image):
    image = image.resize((224, 224))
    img = np.asarray(image).astype(np.float32)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Normalize and reshape
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx = np.argmax(output_data)
    label = class_names[top_idx] if class_names else f"Class {top_idx}"
    confidence = float(output_data[top_idx])
    return label, confidence

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(image)

    st.subheader("Prediction:")
    st.write(f"**Label:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
