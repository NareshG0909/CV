import cv2
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from skimage.feature import hog, graycomatrix, graycoprops
from glob import glob

# --- Load feature dataset and train a classifier ---
def load_traditional_classifier(csv_path, use_pca=False):
    expected_columns = [
        'color_score', 'edge_strength', 'mean_intensity', 'corner_strength', 'hog_mean', 'object_area',
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'entropy',
        'saturation_mean', 'color_std', 'shape_ratio', 'area_bucket'
    ]

    df = pd.read_csv(csv_path)
    if not set(expected_columns).issubset(df.columns):
        print("⚠️ Required columns not found in CSV. Regenerating...")
        extract_features_to_csv('TrashType_Image_Dataset/images/', csv_path)
        df = pd.read_csv(csv_path)

    X = df[expected_columns]
    y = df['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if use_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        joblib.dump(pca, 'results/pca_model.pkl')

    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                        colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    clf.fit(X_scaled, y_encoded)

    y_pred = clf.predict(X_scaled)
    print("\n📊 Classification Report (Training Data):")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

    joblib.dump(clf, 'results/cv_rf_model.pkl')
    joblib.dump(scaler, 'results/feature_scaler.pkl')
    joblib.dump(label_encoder, 'results/label_encoder.pkl')
    print("✅ Model and encoders saved for deployment in /results/")

    # --- Feature Importance Plot ---
    importances = clf.feature_importances_

    if not use_pca:
        feature_names = X.columns
    else:
        feature_names = [f'PC{i+1}' for i in range(len(importances))]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.title("Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/feature_importances.png")
    plt.show()

    return clf, scaler, label_encoder, pca

# --- Feature extraction from cropped object ---
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
    shape_ratio = crop.shape[1] / crop.shape[0]  # width / height

    area_bucket = 0 if area < 2000 else (1 if area < 6000 else 2)
    return [color_score, edge_strength, mean_intensity, corner_strength, hog_mean, area,
            contrast, dissimilarity, homogeneity, energy, correlation, asm, entropy,
            saturation_mean, color_std, shape_ratio, area_bucket]

# --- Extract features from labeled dataset ---
def extract_features_to_csv(input_dir, output_csv):
    data = []
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label in class_dirs:
        image_paths = glob(os.path.join(input_dir, label, '*.*'))
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < 200:
                    continue
                crop = image[y:y+h, x:x+w]
                features = extract_features_from_crop(crop, area)
                data.append(features + [label])

    df = pd.DataFrame(data, columns=[
        'color_score', 'edge_strength', 'mean_intensity', 'corner_strength', 'hog_mean', 'object_area',
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'entropy',
        'saturation_mean', 'color_std', 'shape_ratio', 'area_bucket', 'label']
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Feature extraction complete. Saved to {output_csv}")


    # --- Classify objects inside an image ---
def classify_objects_in_image(image_path, model, scaler, label_encoder, pca=None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"❌ Cannot read image at path: {image_path}")
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        plastic_count = 0

        def get_area_bucket(area):
            if area < 2000:
                return 'Small'
            elif area < 6000:
                return 'Medium'
            else:
                return 'Large'

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 200:
                continue

            crop = image[y:y+h, x:x+w]
            features = extract_features_from_crop(crop, area)
            features_scaled = scaler.transform([features])
            if pca:
                features_scaled = pca.transform(features_scaled)

            proba = model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(proba)
            confidence = np.max(proba)
            label = label_encoder.inverse_transform([pred_class])[0]

            cx = x + w // 2
            cy = y + h // 2

            print(f"🔍 Found {label} (conf={confidence:.2f}) at ({x},{y},{w},{h})")

            cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(orig, f"{label} ({confidence:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if label.lower() == "plastic":
                color = (0, 255, 0) if confidence >= 0.25 else (0, 165, 255)
                mark = "✓" if confidence >= 0.25 else "?"
                cv2.putText(orig, mark, (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                if confidence >= 0.25:
                    plastic_count += 1
                cv2.imwrite(f'results/plastic_debug/{x}_{y}_{w}_{h}.jpg', crop)

            area_bucket = get_area_bucket(area)
            results.append({
                "label": label,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "area": area,
                "area_bucket": area_bucket,
                "confidence": round(confidence, 3)
            })

        return orig, results, plastic_count



if __name__ == "__main__":
        csv_path = 'TrashType_Image_Dataset/results/features1.csv'
        dataset_dir = 'TrashType_Image_Dataset/images/mixed_test_collage_1.jpg'
        image_path = 'results/conveyor_mixed_scenes/conveyor_synthetic_3.jpg'

        if not os.path.exists(csv_path):
            print("⚠️ features1.csv not found. Extracting features from dataset...")
            extract_features_to_csv(dataset_dir, csv_path)

        model, scaler, label_encoder, pca = load_traditional_classifier(csv_path, use_pca=True)
        annotated_img, detections, plastic_count = classify_objects_in_image(image_path, model, scaler, label_encoder, pca)

        print("\n🧠 Detected Objects:")
        for det in detections:
            print(f"{det['label']} at {det['bbox']} (centroid: {det['centroid']}, area: {det['area']}, size: {det['area_bucket']}, confidence: {det['confidence']})")

        print(f"\n✅ Total plastic objects detected: {plastic_count}")

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.title("Plastic Objects Marked with Tick")
        plt.axis('off')
        plt.show()

        output_path = 'results/annotated_output.jpg'
        cv2.imwrite(output_path, annotated_img)
        print(f"\n💾 Annotated image saved to: {output_path}")

        detections_df = pd.DataFrame(detections)
        detections_csv_path = 'results/detections_summary.csv'
        detections_df.to_csv(detections_csv_path, index=False)
        print(f"📄 Detection summary saved to: {detections_csv_path}")
