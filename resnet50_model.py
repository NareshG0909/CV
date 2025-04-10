import os
from PIL import Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import requests
from io import BytesIO
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.layers import Input, Activation, Add, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import UpSampling2D,BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.utils import plot_model

dataset_paths = [
    "/Users/apple/CV/TrashType_Image_Dataset/source/", 
   # "/kaggle/input/extras"
]


# Set to store unique image dimensions for the entire dataset
all_dimensions_set = set()
def load_dataset_images(dataset_paths):
    """
    Loads image file names from a folder.
    
    Args:
        folder_path (str): Path to the image dataset folder.
        max_images (int): Maximum number of images to load (default: 7).

    Returns:
        list: List of image file names.
    """
# Iterate over both datasets
    for dataset_path in dataset_paths:
        garbage_types = os.listdir(dataset_path)

        for garbage_type in garbage_types:
            folder_path = os.path.join(dataset_path, garbage_type)

            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

            # Display the count of images in the current folder
                num_images = len(image_files)
                print(f"{garbage_type} folder contains {num_images} images.")

            # Check dimensions of each image
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    with Image.open(image_path) as img:
                        width, height = img.size
                        channels = len(img.getbands())  # Get number of color channels
                        all_dimensions_set.add((width, height, channels))

# Determine if all images have the same dimensions
    if len(all_dimensions_set) == 1: 
        width, height, channels = all_dimensions_set.pop()
        print(f"\nAll images in the dataset have the same dimensions: {width}x{height} with {channels} color channels.")
    else:
        print("\nThe images in the dataset have different dimensions or color channels.")

def convert_dataset_unique_images(dataset_paths):
# Iterate over both datasets to display images
    for dataset_path in dataset_paths:
        garbage_types = os.listdir(dataset_path)

        for garbage_type in garbage_types:
            folder_path = os.path.join(dataset_path, garbage_type)

        # Verify that the current item is a directory
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

            # Set up subplots
                fig, axs = plt.subplots(1, len(image_files), figsize=(15, 2))

                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        with Image.open(image_path) as img:
                            img = img.convert("RGB")  # Ensure all images are RGB
                            img = img.resize((224, 224))  # Resize to 224x224
                        
                            width, height = img.size
                            channels = len(img.getbands())
                            all_dimensions_set.add((width, height, channels))
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

# Print unique dimensions found
    print("\nUnique image dimensions found in the dataset:")
    for dim in all_dimensions_set:
        print(dim)
def display_dataset_images(dataset_paths,max_images=7):
    data = []
    for dataset_path in dataset_paths:
        garbage_types = os.listdir(dataset_path)

        for garbage_type in garbage_types:
            folder_path = os.path.join(dataset_path, garbage_type)
            # Verify that the current item is a directory
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
    
            # Select the first 5 images
                image_files = image_files[:max_images]

            # Set up subplots
           
                fig, axs = plt.subplots(1, len(image_files), figsize=(15, 2))
                
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(folder_path, image_file)
                    with Image.open(image_path) as img:
                            axs[i].imshow(img)
                            axs[i].axis('off')  # Hide axis
                    
                plt.tight_layout()
                fig.suptitle(f"{garbage_type} ({os.path.basename(dataset_path)})", fontsize=16, y=1.03)
                plt.show()  # Displays the entire figure
# 🔹 Data Preparation
def prepare_dataset_images(dataset_paths):
    """Loads dataset images and creates a DataFrame."""
    data = []
    for dataset_path in dataset_paths:
        garbage_types = os.listdir(dataset_path)
        for garbage_type in garbage_types:
            folder_path = os.path.join(dataset_path, garbage_type)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('jpg', 'jpeg', 'png')):
                        data.append((os.path.join(folder_path, file), garbage_type))

    df = pd.DataFrame(data, columns=['filepath', 'label'])
    print(f"✅ Total images in dataset: {len(df)}")
    print(df['label'].value_counts())  # Print class distribution
    return df

def split_dataset(df):
    """Splits dataset into train, validation, and test sets."""
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    print(f"✅ Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Class distributions in percentages
    overall_distribution = df['label'].value_counts(normalize=True) * 100
    train_distribution = train_df['label'].value_counts(normalize=True) * 100
    val_distribution = val_df['label'].value_counts(normalize=True) * 100
    test_distribution = test_df['label'].value_counts(normalize=True) * 100

# Print distributions
    print("\nClass distribution in the entire dataset:\n", overall_distribution.round(2))
    print('-'*40)
    print("\nClass distribution in the training set:\n", train_distribution.round(2))
    print('-'*40)
    print("\nClass distribution in the validation set:\n", val_distribution.round(2))
    print('-'*40)
    print("\nClass distribution in the test set:\n", test_distribution.round(2))

    return train_df, val_df, test_df 
def resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.summary()
    len(base_model.layers) 
    for i, layer in enumerate(base_model.layers):
        if 140 <= i <= 150:
            print(i, layer.name)
    for layer in base_model.layers[:143]: # include the layer 142
        layer.trainable = False
    # Build the new model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(6, activation='softmax')(x)  # Updated to 10 classes

    transfer_resnet50_model = Model(inputs=base_model.input, outputs=x)

# Compile the model
    transfer_resnet50_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                loss='categorical_crossentropy', 
                                metrics=['accuracy'])
    return transfer_resnet50_model
    
def apply_resnet50(train_df,val_df,test_df):
    

   train_datagen = ImageDataGenerator(
        rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
        zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
        shear_range=0.1, brightness_range=[0.7, 1.3],
        channel_shift_range=20, fill_mode='nearest',
        preprocessing_function=preprocess_input
    )
   val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

   train_generator = train_datagen.flow_from_dataframe(train_df, x_col="filepath", y_col="label",
                                                        target_size=(224, 224), batch_size=32, class_mode='categorical', seed=42)
   val_generator = val_test_datagen.flow_from_dataframe(val_df, x_col="filepath", y_col="label",
                                                         target_size=(224, 224), batch_size=32, class_mode='categorical', seed=42)
   test_generator = val_test_datagen.flow_from_dataframe(test_df, x_col="filepath", y_col="label",
                                                          target_size=(224, 224), batch_size=32, class_mode='categorical', seed=42, shuffle=False)

   print(f"✅ Data Generators Ready: Train={len(train_generator)}, Val={len(val_generator)}, Test={len(test_generator)}")
    

# ✅ Print batch counts
   print(f"Number of batches in train_generator: {len(train_generator)}")
   print(f"Number of batches in val_generator: {len(val_generator)}")
   print(f"Number of batches in test_generator: {len(test_generator)}")

# ✅ Print class indices
   train_generator.class_indices
   return train_generator,val_generator,test_generator

def apply_class_weights(train_df,train_generator):
    """Computes class weights to handle class imbalance."""
    y_numeric = train_df['label'].map(train_generator.class_indices)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_numeric), y=y_numeric)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"✅ Computed Class Weights: {class_weights_dict}")
    return class_weights_dict

def fine_tune(transfer_resnet50_model,class_weights,train_generator,val_generator):
    # Define the callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True, verbose=1)

# Train the model
    num_epochs = 20

    history = transfer_resnet50_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator), 
        epochs=num_epochs,
        validation_data=val_generator, 
        validation_steps=len(val_generator),
        class_weight=class_weights,  # Applying computed class weights
        callbacks=[reduce_lr, early_stopping]
    )
    transfer_resnet50_model.save("resnet_model_1.h5")
    return history

def plot_learning_curves(history, start_epoch=1):
    """
    Plot training and validation loss and accuracy curves.

    Parameters:
    - history: Training history (output from model.fit()).
    - start_epoch: Epoch from which to start plotting. Default is 1.
    """

    # Convert history.history dict to a pandas DataFrame
    df = pd.DataFrame(history.history)

    # Plot curves from specified epoch onwards
    df = df.iloc[start_epoch - 1:]

    # Seaborn styling
    sns.set(rc={'axes.facecolor': '#f0f0fc'}, style='darkgrid')

    plt.figure(figsize=(15, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=df.index, y=df['loss'], color='royalblue', label='Train Loss')
    sns.lineplot(x=df.index, y=df['val_loss'], color='orangered', linestyle='--', label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=df.index, y=df['accuracy'], color='royalblue', label='Train Accuracy')
    sns.lineplot(x=df.index, y=df['val_accuracy'], color='orangered', linestyle='--', label='Validation Accuracy')
    plt.title('Accuracy Evolution')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()
def evaluate_model_performance(model, test_generator, class_labels):
    """
    Evaluate the model's performance on the test set, print accuracy, and classification report.

    Parameters:
    - model: The trained model.
    - test_generator: Test data generator.
    - class_labels: List of class names.
    """
    
    """Evaluates model on test set and plots confusion matrix."""
    true_labels = test_generator.classes
    predictions = model.predict(test_generator, steps=len(test_generator))
    predicted_labels = np.argmax(predictions, axis=1)
    
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print(classification_report(true_labels, predicted_labels, target_names=class_labels))
    
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_random_predictions(test_generator, model, class_labels, num_samples=10):
    """
    Selects random images from the test set, predicts them, and visualizes results with confidence scores.

    Parameters:
    - test_generator: Keras ImageDataGenerator for testing.
    - model: Trained deep learning model.
    - class_labels: List of class names.
    - num_samples: Number of random images to display.
    """
    test_generator.reset()
    x_batch, y_batch = next(iter(test_generator))  # Corrected `.next()`
    
    random_indices = np.random.choice(len(x_batch), size=num_samples, replace=False)
    true_labels = np.argmax(y_batch[random_indices], axis=1)

    predictions = model.predict(x_batch[random_indices])
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1) * 100
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_batch[idx].astype("uint8"))
        plt.axis("off")
        plt.title(f"Pred: {class_labels[predicted_labels[i]]}")
        plt.title(f"True: {class_labels[true_labels[i]]}\n"
                  f"Pred: {class_labels[predicted_labels[i]]}\n"
                  f"Conf: {confidence_scores[i]:.2f}%", fontsize=9)
    
    plt.tight_layout()
    plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

def get_model_summary(model, test_generator, model_name):
    true_labels = test_generator.classes
    predictions = model.predict(test_generator, steps=len(test_generator))
    predicted_labels = np.argmax(predictions, axis=1)

    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4)
    }

def main():
    print("✅ Starting main function...")  # Debug print

    print("Loading dataset images and checking dimensions...\n")
    load_dataset_images(dataset_paths)
    
    print("\nConverting dataset images to uniform size and format...\n")
    
    print("\nDisplaying test images")
    
    print("\n Dataset preparation ")
    df = prepare_dataset_images(dataset_paths)
    
    print("✅ Dataset preparation complete.")

    print("Splitting dataset into train, validation, and test sets")
    train_df, val_df, test_df = split_dataset(df)
    
    print("✅ Dataset splitting complete.")

    print("Applying ResNet50")
    train_generator, val_generator, test_generator = apply_resnet50(train_df, val_df, test_df)

    print("✅ ResNet50 applied successfully.")
    
    class_labels = list(train_generator.class_indices.keys())
    
    print("Computing class weights")
    class_weights = apply_class_weights(train_df, train_generator)

    print("✅ Class weights computed.")

    print("Initializing ResNet50 model")
    transfer_resnet50_model = resnet50()

    print("✅ ResNet50 model initialized.")

    print("Fine-tuning the model")
    history = fine_tune(transfer_resnet50_model, class_weights, train_generator, val_generator)

    print("✅ Fine-tuning complete.")

    print("Plotting learning curves")
    plot_learning_curves(history, start_epoch=1)

    print("Evaluating model performance")
    confidence_scores = evaluate_model_performance(transfer_resnet50_model, test_generator, class_labels)

    print(confidence_scores)
    plot_random_predictions(test_generator, transfer_resnet50_model, class_labels, num_samples=10)
    print("✅ Model evaluation complete.")

     # Model summary output
    summary = get_model_summary(transfer_resnet50_model, test_generator, "ResNet50 (Transfer Learning)")
    summary_df = pd.DataFrame([summary])
    print("Model Performance Summary:")
    print(summary_df.to_string(index=False))

    plot_random_predictions(test_generator, transfer_resnet50_model, class_labels, num_samples=10)
    print("Model evaluation complete.")
  

# Make sure this part is at the same indentation level
if __name__ == "__main__":
    main()
