import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

IMG_SIZE = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan to classify the tumor type using ResNet50.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_resnet50_final.keras")


def preprocess(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        processed = preprocess(image)
        preds = model.predict(processed)[0]
        predicted = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

    st.markdown(f"### Prediction: **{predicted}**")
    st.markdown(f"Confidence: **{confidence*100:.1f}%**")

    st.markdown("#### Class Probabilities")
    for name, prob in zip(CLASS_NAMES, preds):
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

    st.warning("⚠️ This tool is for research purposes only. Not a medical diagnosis.")
