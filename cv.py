import cv2
import numpy as np
import os
import json
from glob import glob

# --- Parameters ---
data_dir = '/Users/apple/CV/TrashType_Image_Dataset/paper'
output_dir = '/Users/apple/CV/TrashType_Image_Dataset/results/'
os.makedirs(output_dir, exist_ok=True)

# --- Color ranges for categories (tweak as needed) ---
CATEGORY_COLORS = {
    'plastic': ([90, 50, 50], [130, 255, 255]),       # blue-ish
    'cardboard': ([10, 50, 50], [25, 255, 200]),      # brown
    'metal': ([0, 0, 50], [180, 50, 180]),            # gray/silver
    'glass': ([35, 30, 50], [85, 255, 255]),          # green/clear
    'paper': ([0, 0, 180], [180, 30, 255])            # white/pale
}

# --- Helper: Convert NumPy types to native Python types ---
def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj

# --- Classification Function ---
def classify_object(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts = {}

    for category, (lower, upper) in CATEGORY_COLORS.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        counts[category] = cv2.countNonZero(mask)

    edges = cv2.Canny(gray, 50, 150)
    edge_strength = np.sum(edges) / 255

    mean_intensity = np.mean(gray)

    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    corner_strength = np.sum(corners > 0.01 * corners.max())

    features = {
        'color_score': int(max(counts.values())),
        'edge_strength': float(edge_strength),
        'mean_intensity': float(mean_intensity),
        'corner_strength': int(corner_strength)
    }

    if edge_strength > 1100 and corner_strength > 100:
        predicted_class = 'metal'
    elif mean_intensity > 185:
        predicted_class = 'paper'
    elif counts['glass'] > 800:
        predicted_class = 'glass'
    elif counts['cardboard'] > counts['plastic']:
        predicted_class = 'cardboard'
    else:
        predicted_class = max(counts, key=counts.get)

    return predicted_class, features

# --- Load image paths ---
image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    image_paths.extend(glob(os.path.join(data_dir, ext)))

print(f"Found {len(image_paths)} image(s)")

# --- Classification and Results Storage ---
labels = {}
correct_count = 0
wrong_count = 0

for img_path in image_paths:
    filename = os.path.basename(img_path)
    try:
        actual_label = filename.split('_')[0].lower()
    except:
        print(f"[Warning] Could not parse label from: {filename}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warning] Couldn't read image: {img_path}")
        continue

    resized = cv2.resize(img, (128, 128))
    predicted_label, feature_summary = classify_object(resized)
    correct = predicted_label == actual_label

    if correct:
        correct_count += 1
    else:
        wrong_count += 1

    labels[filename] = {
        'actual': actual_label,
        'predicted': predicted_label,
        'correct': correct,
        'features': feature_summary
    }

    # Annotate and save image
    annotated = resized.copy()
    color = (0, 255, 0) if correct else (0, 0, 255)
    cv2.putText(annotated, f"Pred: {predicted_label}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(annotated, f"True: {actual_label}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, annotated)

# --- Save labels.json with safe conversion ---
json_path = os.path.join(output_dir, "labels.json")
with open(json_path, 'w') as f:
    json.dump(convert_to_native(labels), f, indent=4)

# --- Final Summary ---
print(f"\n✅ Classification complete:")
print(f"Correct predictions: {correct_count}")
print(f"Misclassifications: {wrong_count}")
print(f"Accuracy: {correct_count / (correct_count + wrong_count):.2%}")
print(f"Labels saved to: {json_path}")
