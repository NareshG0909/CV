from ultralytics import YOLO

# Create model from base config
model = YOLO('yolov8n.yaml')  # or yolov8s.yaml for more accuracy

# Train with your data
model.train(
    data='yolo_dataset/data.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    name='trash_classifier_6class',
    project='runs/detect'
)
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/trash_classifier_6class/weights/best.pt')

# Run inference
results = model.predict(source='/Users/apple/Downloads/yolotest1.jpg', show=True, conf=0.25)