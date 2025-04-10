import kagglehub
import shutil


# Download latest version
path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")

print("Path to dataset files:", path)


# Define source and destination paths
source_dir = "/Users/apple/.cache/kagglehub/datasets/farzadnekouei/trash-type-image-dataset/versions/1/TrashType_Image_Dataset"
destination_dir = "/Users/apple/CV/TrashType_Image_Dataset"

# Copy the entire directory
shutil.copytree(source_dir, destination_dir)
