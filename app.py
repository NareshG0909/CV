# ✅ Revised Streamlit App Code with Fixes for CVClassifier model (PCA loading issue resolved)

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
import cv2
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage.feature import hog, graycomatrix, graycoprops

# --- Download helper
def download_model_from_drive(file_id, output_path):
    folder = os.path.dirname(output_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# --- Load YOLO model
@st.cache_resource
def load_yolo():
    yolo_path = "models/yolo_model_ne.pt"
    download_model_from_drive("1702UCd6U6c0E1E23m5HKR43oME55cTmn", yolo_path)
    return YOLO(yolo_path)

# --- Load ResNet/SegNet or CVClassifier model
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
        download_model_from_drive("1fIbYdzdoCnxpUZP2QknGPTDDKENXeCog", model_path)
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

    return None  # for CVClassifier which uses joblib

# --- Class labels
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- Preprocess uploaded image for deep learning
def preprocess_uploaded_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# --- CV feature extractor
def extract_features_from_crop(crop, area):
    crop = cv2.resize(crop, (128, 128))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    color_score = np.mean(crop)
    edge_strength = np.mean(cv2.Canny(crop, 100, 200))
    mean_intensity = np.mean(gray)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corner_strength = len(corners) if corners is not None else 0
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, feature_vector=True)
    hog_mean = np.mean(hog_features)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    glcm_prob = glcm.astype(np.float64) / np.sum(glcm)
    entropy = -np.sum(glcm_prob * np.log2(glcm_prob + 1e-10))

    saturation_mean = np.mean(hsv[:, :, 1])
    color_std = np.std(crop)
    shape_ratio = crop.shape[1] / crop.shape[0]
    area_bucket = 0 if area < 2000 else (1 if area < 6000 else 2)

    return [color_score, edge_strength, mean_intensity, corner_strength, hog_mean, area,
            contrast, dissimilarity, homogeneity, energy, correlation, asm, entropy,
            saturation_mean, color_std, shape_ratio, area_bucket]

# --- CV-based classification
def classify_image_with_cv(uploaded_file):
    # Download all CVClassifier joblib files from Google Drive
    download_model_from_drive("1vN4C6NE3vjBZL7bViKZxW4_QPeZBnIra", "results/feature_scaler.pkl")
    download_model_from_drive("1fsKRyYjVw1uyAjVXp1mt8TYWaZXVRLYa", "results/label_encoder.pkl")
    download_model_from_drive("1FZxVYV4trBEo9pdWRAtUpkElkxUeL3ZK", "results/cv_rf_model.pkl")
    download_model_from_drive("1ZwKhqFCPAhWqx5dHRWQY0WB46UpgTxLD", "results/pca_model.pkl")

    model = joblib.load("results/cv_rf_model.pkl")
    scaler = joblib.load("results/feature_scaler.pkl")
    label_encoder = joblib.load("results/label_encoder.pkl")
    pca = joblib.load("results/pca_model.pkl") if os.path.exists("results/pca_model.pkl") else None

    img = Image.open(uploaded_file).convert("RGB")
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    predictions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1000:
            continue
        crop = image[y:y+h, x:x+w]
        features = extract_features_from_crop(crop, area)[:13]  # Trim to match scaler
        features = np.asarray(features).reshape(1, -1)
        if features.shape[1] != scaler.n_features_in_:
            st.warning(f"Feature dimension mismatch: expected {scaler.n_features_in_}, got {features.shape[1]}")
            continue
        features_scaled = scaler.transform(features)
        if pca is not None and features_scaled.shape[1] == pca.n_features_in_:
            features_scaled = pca.transform(features_scaled)
        proba = model.predict_proba(features_scaled)[0]
        pred_class = np.argmax(proba)
        label = label_encoder.inverse_transform([pred_class])[0]
        confidence = np.max(proba)
        predictions.append((label, confidence))
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(output, f"{label} ({confidence:.2f})", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return output, predictions

# --- Streamlit UI
def main():
    st.title("♻️ Garbage Classification App")
    st.write("Upload an image and choose a model to classify or detect trash.")

    model_choice = st.selectbox("Choose a model", [
        "ResNet", "SegNet", "ResNetsegNet", "ResNetSegNetFinetuned", "YOLOv8", "CVClassifier"
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

        elif model_choice == "CVClassifier":
            annotated_img, predictions = classify_image_with_cv(uploaded_file)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="CV Detected Objects")
            for label, confidence in predictions:
                st.write(f"- {label}: {confidence:.2%}")

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
