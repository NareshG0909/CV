import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from ultralytics import YOLO

# Load YOLOv8 model once
@st.cache_resource
def load_yolo():
    return YOLO("TrashType_Image_Dataset/model/yolo_model.pt")  # Updated pathTrashType_Image_Dataset/model/resnet_model.h5

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
        return load_model("TrashType_Image_Dataset/model/resnet_model.h5")
    elif model_name == "SegNet":
        return load_model("TrashType_Image_Dataset/model/segnet_model.h5", custom_objects=custom_objects)
    elif model_name == "ResNetsegNet":
        return load_model("TrashType_Image_Dataset/model/resnet_segnet_model.h5", custom_objects=custom_objects)
    elif model_name == "ResNetSegNetFinetuned":
        return load_model("TrashType_Image_Dataset/model/resnet_segnet_model_finetuned.h5", custom_objects=custom_objects)

# Class labels
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Image preprocessor for CNN models
def preprocess_uploaded_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Basic CV-based classifier (demo only)
def classify_with_cv(image):
    gray = np.mean(np.array(image.convert("L")))
    if gray < 85:
        return "Metal", 75.0
    elif gray < 130:
        return "Plastic", 65.0
    else:
        return "Paper", 60.0

# Streamlit app logic
def main():
    st.title("♻️ Garbage Classification App")
    st.write("Upload an image and choose a model to classify or detect trash.")

    model_choice = st.selectbox("Choose a model", [
        "ResNet", "SegNet", "ResNetsegNet", "ResNetSegNetFinetuned", "YOLOv8", "Basic CV Classifier"
    ])

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_column_width=True)

        st.markdown("### 🔍 Model Output")
        if model_choice == "YOLOv8":
            yolo_model = load_yolo()
            results = yolo_model.predict(image,show=False, conf=0.01)
            result_img = results[0].plot()
            st.image(result_img, caption="YOLOv8 Detections")

            # Display class labels and confidences
            detections = results[0].boxes
            if detections is not None and len(detections.cls) > 0:
                st.markdown("**Detected Objects:**")
                for cls_id, conf in zip(detections.cls.tolist(), detections.conf.tolist()):
                    class_name = CLASS_LABELS[int(cls_id)] if int(cls_id) < len(CLASS_LABELS) else f"Class {int(cls_id)}"
                    st.write(f"- {class_name}: {conf:.2%}")
            else:
                st.write("No objects detected.")

        elif model_choice == "Basic CV Classifier":
            label, confidence = classify_with_cv(image)
            st.markdown(f"**Predicted:** `{label}` with confidence `{confidence:.1f}%`")

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
