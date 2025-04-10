import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from ultralytics import YOLO

# Function to download models from Google Drive
def download_model_from_drive(file_id, output_path):
    folder = os.path.dirname(output_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)  # ✅ Create 'models/' if it doesn't exist

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Load YOLOv8 model once
@st.cache_resource
def load_yolo():
    yolo_path = "models/yolo_model_ne.pt"
    download_model_from_drive("1Zm9ttfL-YdcP6JcMUELRUxZh6oKcy_qC", yolo_path)  # Replace with your actual file ID
    return YOLO(yolo_path)

# Load ResNet/SegNet/ResNetSegNet models
@st.cache_resource
def load_trained_model_variant(model_name):
    class_weights = np.array([1.0, 2.0, 1.5, 1.2, 1.8, 1.0])

    def weighted_categorical_crossentropy(weights):
        weights = K.variable(weights)
        def loss(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * K.log(y_pred) * weights
            return -K.sum(loss, -1)
        return loss

    custom_loss = weighted_categorical_crossentropy(class_weights)
    custom_objects = {'loss': custom_loss}

    if model_name == "ResNet":
        model_path = "models/resnet_model.h5"
        download_model_from_drive("1PiN8NxiPARrOgsJapdKPgc9x-qDlSdeH", model_path)
        return load_model(model_path)

    elif model_name == "SegNet":
        model_path = "models/segnet_model.h5"
        download_model_from_drive("1eM4VffGW5GwLX4GeSyZT4I9sbmOeLq76", model_path)
        return load_model(model_path, custom_objects=custom_objects)

    elif model_name == "ResNetsegNet":
        model_path = "models/resnet_segnet_model.h5"
        download_model_from_drive("1W6NuzGRY33xJPBXdlTPWUxd9CQ-EVXAI", model_path)
        return load_model(model_path, custom_objects=custom_objects)

    elif model_name == "ResNetSegNetFinetuned":
        model_path = "models/resnet_segnet_model_finetuned.h5"
        download_model_from_drive("1ZF-fdUWhnpMQA-GrdQwrMwFDGBfHOTlR", model_path)
        return load_model(model_path, custom_objects=custom_objects)

# Class labels
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Image preprocessor for CNN models
def preprocess_uploaded_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Streamlit app logic
def main():
    st.title("♻️ Garbage Classification App")
    st.write("Upload an image and choose a model to classify or detect trash.")

    model_choice = st.selectbox("Choose a model", [
        "ResNet", "SegNet", "ResNetsegNet", "ResNetSegNetFinetuned", "YOLOv8"
    ])

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_column_width=True)

        st.markdown("### 🔍 Model Output")
        if model_choice == "YOLOv8":
            yolo_model = load_yolo()
            results = yolo_model.predict(image, show=False, conf=0.01)
            result_img = results[0].plot()
            st.image(result_img, caption="YOLOv8 Detections")

            detections = results[0].boxes
            if detections is not None and len(detections.cls) > 0:
                st.markdown("**Detected Objects:**")
                for cls_id, conf in zip(detections.cls.tolist(), detections.conf.tolist()):
                    class_name = CLASS_LABELS[int(cls_id)] if int(cls_id) < len(CLASS_LABELS) else f"Class {int(cls_id)}"
                    st.write(f"- {class_name}: {conf:.2%}")
            else:
                st.write("No objects detected.")

        else:
            model = load_trained_model_variant(model_choice)
            processed_image = preprocess_uploaded_image(image)
            prediction = model.predict(processed_image)
            predicted_label = CLASS_LABELS[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            st.markdown(f"**Predicted:** `{predicted_label}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

if __name__ == "__main__":
    main()
