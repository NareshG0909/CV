# ✅ ResNet50 Trash Classification - Full Pipeline with Fixes
import os
from PIL import Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Dataset path
dataset_paths = ["/Users/apple/CV/TrashType_Image_Dataset/images/"]

# Utility logging
log = lambda msg: print(f"\n🧠 {msg}")

# Step 1: Load and Validate Dataset
def prepare_dataset_images(dataset_paths):
    data = []
    for dataset_path in dataset_paths:
        garbage_types = os.listdir(dataset_path)
        for garbage_type in garbage_types:
            if not garbage_type.strip() or garbage_type.startswith('.'):
                continue
            folder_path = os.path.join(dataset_path, garbage_type)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                if not image_files:
                    continue
                for file in image_files:
                    data.append((os.path.join(folder_path, file), garbage_type))

    df = pd.DataFrame(data, columns=['filepath', 'label'])
    print(f"✅ Total images in dataset: {len(df)}")
    if df.empty:
        raise ValueError("❌ Dataset is empty. Check paths or image formats.")
    print(df['label'].value_counts())
    return df

def split_dataset(df):
    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column.")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:143]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(6, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def apply_resnet50(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                       zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
                                       shear_range=0.1, brightness_range=[0.7, 1.3],
                                       channel_shift_range=20, fill_mode='nearest',
                                       preprocessing_function=preprocess_input)

    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col="filepath", y_col="label",
                                                        target_size=(224, 224), batch_size=32,
                                                        class_mode='categorical', seed=42)

    val_generator = val_test_datagen.flow_from_dataframe(val_df, x_col="filepath", y_col="label",
                                                         target_size=(224, 224), batch_size=32,
                                                         class_mode='categorical', seed=42)

    test_generator = val_test_datagen.flow_from_dataframe(test_df, x_col="filepath", y_col="label",
                                                          target_size=(224, 224), batch_size=32,
                                                          class_mode='categorical', seed=42, shuffle=False)

    return train_generator, val_generator, test_generator

def apply_class_weights(train_df, train_generator):
    y_numeric = train_df['label'].map(train_generator.class_indices)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_numeric), y=y_numeric)
    return {i: w for i, w in enumerate(class_weights)}

def fine_tune(model, class_weights, train_gen, val_gen):
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=20,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        class_weight=class_weights,
        callbacks=callbacks
    )
    model.save("resnet_model_1.h5")
    return history

def main():
    log("Preparing dataset...")
    df = prepare_dataset_images(dataset_paths)
    train_df, val_df, test_df = split_dataset(df)

    log("Creating generators...")
    train_gen, val_gen, test_gen = apply_resnet50(train_df, val_df, test_df)
    class_labels = list(train_gen.class_indices.keys())

    log("Computing class weights...")
    class_weights = apply_class_weights(train_df, train_gen)

    log("Initializing model...")
    model = resnet50()

    log("Training model...")
    history = fine_tune(model, class_weights, train_gen, val_gen)

    log("All done. Model saved as resnet_model_1.h5")

if __name__ == "__main__":
    main()
