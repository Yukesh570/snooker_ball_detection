import os
from ultralytics import YOLO
import cv2

# Directory and file paths
IMAGES_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
image_path = os.path.join(IMAGES_DIR, '7.jpg')
image_path_out = '{}_out.jpg'.format(image_path)

# Read the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

H, W, _ = image.shape

# Load the YOLO model
model_path = os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolo_8', 'runs', 'detect', 'train6  ', 'weights', 'last.pt')
model = YOLO(model_path)  # Load the custom model

# Perform object detection
results = model(image)[0]

# Threshold for displaying bounding boxes
threshold = 0.5

# Draw bounding boxes on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the processed image
cv2.imwrite(image_path_out, image)

print(f"Processed image saved to {image_path_out}")