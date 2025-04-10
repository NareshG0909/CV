
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/trash_classifier_6class/weights/best.pt')

# Run inference
results = model.predict(source='TrashType_Image_Dataset/input/sample.jpg', show=True, conf=0.01)