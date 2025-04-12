import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load model ---
print("Loading trained ResNet50 model")
model_path = "resnet_model_1.h5"
model = load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# --- Prepare test dataset from image files ---
print("Locating image files in test folder")
test_image_dir = "yolo_dataset/images/val"  # Update this path as needed
test_images = glob(os.path.join(test_image_dir, "**", "*.jpg"), recursive=True)
test_images += glob(os.path.join(test_image_dir, "**", "*.jpeg"), recursive=True)
test_images += glob(os.path.join(test_image_dir, "**", "*.png"), recursive=True)

if not test_images:
    raise FileNotFoundError("❌ No image files found in the specified test_images folder.")

# Infer labels from folder names
image_label_pairs = [(img, os.path.basename(os.path.dirname(img))) for img in test_images]
test_df = pd.DataFrame(image_label_pairs, columns=["filepath", "label"])
print(f"✅ Found {len(test_df)} test images.")

# Ensure all classes are known to the model
all_classes = sorted(list(set(test_df['label'].values)))

# --- Load test generator ---
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Make predictions ---
print("Generating predictions on test data")
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print("✅ Predictions complete.")

# --- Evaluation metrics ---
print("Evaluating model performance")
acc = accuracy_score(true_classes, pred_classes)
f1 = f1_score(true_classes, pred_classes, average='weighted')

print("\n🧠 ResNet50 Model Evaluation on Test Data:")
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("\n📊 Classification Report:")
print(classification_report(true_classes, pred_classes, labels=np.arange(len(class_labels)), target_names=class_labels))

# --- Confusion Matrix ---
print("Generating confusion matrix")
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print("✅ Evaluation complete.")
