import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import (UpSampling2D, Dropout, BatchNormalization, Conv2D,
                                     Add, Input, Activation, Dense, MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image

# Define class labels (adjust based on your dataset)
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Custom loss function: weighted categorical crossentropy
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of categorical crossentropy.
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

# Load model based on user selection
@st.cache_resource
def load_trained_model_variant(model_name):
    # Define class weights used during training (replace with actual if needed)
    class_weights = np.array([1.0, 2.0, 1.5, 1.2, 1.8, 1.0])  # Adjust as per your training
    custom_loss = weighted_categorical_crossentropy(class_weights)

    # Define custom objects required by SegNet and ResNetSegNet-based models
    custom_objects = {
        'UpSampling2D': UpSampling2D,
        'Dropout': Dropout,
        'BatchNormalization': BatchNormalization,
        'Conv2D': Conv2D,
        'Add': Add,
        'Input': Input,
        'Activation': Activation,
        'Dense': Dense,
        'MaxPooling2D': MaxPooling2D,
        'GlobalAveragePooling2D': GlobalAveragePooling2D,
        'loss': custom_loss  # Include custom loss function
    }

    if model_name == "ResNet":
        return load_model("resnet_model.h5")
    elif model_name == "SegNet":
        return load_model("segnet_model.h5", custom_objects=custom_objects)
    elif model_name == "ResNetsegNet":
        return load_model("resnet_segnet_model.h5", custom_objects=custom_objects)
    else:  # ResNetSegNetFinetuned
        return load_model("resnet_segnet_model_finetuned.h5", custom_objects=custom_objects)

# Preprocess the image
def preprocess_uploaded_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Main Streamlit app
def main():
    st.title("♻️ Garbage Classification")
    st.write("Upload an image and compare predictions from different models.")

    # Model selector
    model_name = st.selectbox("Choose a Model", ["ResNet", "SegNet", "ResNetsegNet", "ResNetSegNetFinetuned"])
    model = load_trained_model_variant(model_name)

    # Image uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_uploaded_image(image)
        prediction = model.predict(processed_image)
        predicted_label = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display result
        st.markdown(f"### 🧠 Predicted: `{predicted_label}`")
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

if __name__ == "__main__":
    main()
