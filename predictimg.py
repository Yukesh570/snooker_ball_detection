import os
from ultralytics import YOLO
import cv2

# Directory and file paths
IMAGES_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
image_path = os.path.join(IMAGES_DIR, 'images.JPG')
image_path_out = '{}_out.JPG'.format(image_path)

# Read the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

# Load the YOLO model
model_path = os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolo_8', 'runs', 'detect', 'train10', 'weights', 'last.pt')
model = YOLO(model_path)  # Load the custom model

# Perform object detection
results = model.predict(image)

# Threshold for displaying bounding boxes
threshold = 0.5

# Draw bounding boxes on the image
for result in results:
    for box in result.boxes:
        # Extract box attributes
        x1, y1, x2, y2 = box.xyxy[0]
        score = box.conf[0]
        class_id = box.cls[0]

        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            class_name = model.names[int(class_id)].upper() if hasattr(model, 'names') else 'CLASS'
            cv2.putText(image, class_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

# Save the processed image
cv2.imwrite(image_path_out, image)

print(f"Processed image saved to {image_path_out}")
