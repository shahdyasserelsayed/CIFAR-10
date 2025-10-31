import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds
import os

# --- Model Path ---
MODEL_PATH = r"C:\Users\starr\OneDrive\Desktop\DEPI ASSIGNMENTS\CIFAR-10 Project\saved_model\cifar10_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# --- CIFAR-10 Labels ---
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Streamlit UI ---
st.title("🖼️ CIFAR-10 Image Classification Dashboard")
st.markdown("Upload an image below to classify it using the trained CNN model.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload a CIFAR-10 image (32x32 or larger)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((32, 32))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # --- Plot confidence bar chart ---
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0])
    plt.xticks(rotation=45)
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence for Each Class")
    st.pyplot(fig)

st.markdown("---")
st.header("📊 Explore CIFAR-10 Samples")

# --- Load some CIFAR-10 samples ---
data, info = tfds.load("cifar10", split="test[:12]", as_supervised=True, with_info=True)
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for i, (img, label) in enumerate(data.take(12)):
    ax = axes[i // 4, i % 4]
    ax.imshow(img)
    ax.set_title(CLASS_NAMES[int(label)])
    ax.axis("off")
st.pyplot(fig)
