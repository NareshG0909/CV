import cv2
import numpy as np
import os
import csv
from glob import glob

# --- Parameters ---
data_dir = '/Users/apple/CV/TrashType_Image_Dataset/source'
csv_output = '/Users/apple/CV/TrashType_Image_Dataset/results/features.csv'

CATEGORY_COLORS = {
    'plastic': ([90, 50, 50], [130, 255, 255]),
    'cardboard': ([10, 50, 50], [25, 255, 200]),
    'metal': ([0, 0, 50], [180, 50, 180]),
    'glass': ([35, 30, 50], [85, 255, 255]),
    'paper': ([0, 0, 180], [180, 30, 255])
}

def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    return [
        int(max(counts.values())),
        float(edge_strength),
        float(mean_intensity),
        int(corner_strength)
    ]

# --- Write to CSV ---
with open(csv_output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label', 'color_score', 'edge_strength', 'mean_intensity', 'corner_strength'])

    # Search recursively inside each subfolder
    for path in glob(os.path.join(data_dir, '*', '*.*')):
        filename = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path)).lower()  # folder name as label
        img = cv2.imread(path)
        if img is None:
            print(f"[Warning] Could not read image: {filename}")
            continue
        try:
            img = cv2.resize(img, (128, 128))
        except Exception as e:
            print(f"[Error] Resize failed for {filename}: {e}")
            continue
        features = extract_features(img)
        writer.writerow([filename, label] + features)

print("✅ Features saved to:", csv_output)
