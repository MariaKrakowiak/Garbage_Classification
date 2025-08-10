import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------ Configuration ------------------ #
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_INPUT_SHAPE = (128, 128, 3)
MODEL_PATH = "../model/best_model.h5"


# ------------------ Model loading ------------------ #
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the MobileNetV2 model architecture and weights."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=MODEL_INPUT_SHAPE,
        weights=None
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.load_weights(MODEL_PATH)
    return model


# ------------------ Image preprocessing ------------------ #
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare image for model prediction."""
    image = image.convert("RGB")
    image = image.resize(MODEL_INPUT_SHAPE[:2])
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


# ------------------ Prediction and display ------------------ #
def predict_and_display(model, image: Image.Image):
    """Predict class and display results with confidence bars."""
    processed_image = preprocess_image(image)

    with st.spinner('Classifying...'):
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = predictions[0][predicted_idx]

    st.markdown(
        f"<div style='text-align: center; font-size: 24px'>Prediction: <strong>{predicted_class.upper()}</strong></div>",
        unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; font-size: 24x'><b>Confidence:</b> {confidence:.2%}</div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align: center; font-size: 24px'>Confidence per class:</div>",
        unsafe_allow_html=True)
    sorted_confidences = sorted(zip(CLASS_NAMES, predictions[0]),
                                key=lambda x: x[1], reverse=True)

    for cls, score in sorted_confidences:
        bar_length = int(score * 100)
        bar = f"""
        <div style='background-color:#eee; border-radius:5px; width:100%; height:20px; margin-bottom:5px;'>
          <div style='background-color:#4CAF50; width:{bar_length}%; height:20px; border-radius:5px;'></div>
        </div>
        """
        label = f"""
        <div style='text-align:center; margin-bottom:5px; font-size:18px'>
            <strong>{cls.capitalize()}</strong>: {score:.2%}
        </div>
        """
        st.markdown(label, unsafe_allow_html=True)
        st.markdown(bar, unsafe_allow_html=True)


# ------------------ Application layout ------------------ #
def main():
    st.set_page_config(page_title="Waste Wise App", layout="centered")

    st.sidebar.header("How to use")
    st.sidebar.write(
        """
        1. Upload a photo of waste (jpg/jpeg/png).
        2. The model classifies it into one of 6 categories.
        3. See prediction and confidence scores below.
        """
    )

    st.markdown("<h1 style='text-align: center;'>Waste Classification Application</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h3 style="text-align:center; color:#2e7d32;">Why use this waste classification app?</h3>
        <p style="font-size:16px; line-height:1.5; text-align:justify; color:#333;">
        Sorting your garbage correctly can feel confusing — but it’s super important for keeping our planet clean! This app helps <strong>you</strong> easily identify what type of waste you have, so you know exactly how to dispose of it.
        </p>
        <ul style="font-size:16px; color:#333;">
            <li><strong>Save time and hassle:</strong> Just upload a photo and the app tells you whether it’s cardboard, glass, metal, paper, plastic or trash.</li>
            <li><strong>Help the environment:</strong> Proper sorting means more materials get recycled and less ends up in landfills or polluting nature.</li>
            <li><strong>Make a difference:</strong> Small changes in how you throw away waste add up to a big positive impact on your community and the planet.</li>
            <li><strong>Learn as you go:</strong> The app helps you become a pro at recycling without having to memorize complicated rules.</li>
        </ul>
        <p style="text-align:center; font-weight:bold; color:#2e7d32;">
        It’s quick, easy and a smart way to be eco-friendly every day!
        </p>
        </div>
    """, unsafe_allow_html=True)


    st.markdown("""
           <div style='text-align: center; font-size: 18px'>
           Upload an image to classify it into one of the following categories:
           """, unsafe_allow_html=True)
    st.markdown("""
           <div style='text-align: center; font-size: 18px'>
           <strong>Cardboard, Glass, Metal, Paper, Plastic, Trash</strong>
           """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        model = load_model()
        predict_and_display(model, image)


if __name__ == "__main__":
    main()
