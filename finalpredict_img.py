import os
from ultralytics import YOLO
import cv2

# Directory and file paths
IMAGES_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
image_path = os.path.join(IMAGES_DIR, 'triangle.jpg')  # Change to the specific image you want to process
model_paths = [
    os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolofrominternet', 'runs', 'detect', 'train5', 'weights', 'last.pt'),
    os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolo_8', 'runs', 'detect', 'train10', 'weights', 'last.pt')
]

def process_image(image_path, model_paths):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}.")
        return

    # Threshold for displaying bounding boxes
    threshold = 0.5

    for model_path in model_paths:
        # Load the YOLO model
        model = YOLO(model_path)  # Load the custom model

        # Perform object detection
        results = model.predict(image)

        # Draw bounding boxes on the image
        for result in results:
            for box in result.boxes:
                # Extract box attributes
                x1, y1, x2, y2 = box.xyxy[0]
                score = box.conf[0]
                class_id = box.cls[0]

                if score > threshold:
                    # Draw the bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Define the text and get its width and height
                    class_name = model.names[int(class_id)].upper() if hasattr(model, 'names') else 'CLASS'
                    (label_width, label_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                    # Ensure the text fits within the bounding box
                    if y1 + label_height + baseline < y2:
                        # Place text inside the bounding box at the top-left corner
                        cv2.putText(image, class_name, (int(x1), int(y1 + label_height + baseline)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        # If not enough space inside, place it above the box
                        cv2.putText(image, class_name, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Save the processed image
    image_path_out = '{}_out_both_det.jpg'.format(image_path)
    cv2.imwrite(image_path_out, image)

    print(f"Processed image saved to {image_path_out}")

# Process the specified image with both models
process_image(image_path, model_paths)
