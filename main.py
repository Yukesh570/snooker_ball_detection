# This is a sample Python script.
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


# Use the model
results = model.train(data="config.yaml", epochs=50)  # train the model
