from ultralytics import YOLO

# Since best.pt is inside background_removal/models
model = YOLO("models/best.pt")
