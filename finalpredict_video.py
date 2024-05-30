# import os
# from ultralytics import YOLO
# import cv2
#
# # Directory and file paths
# # VIDEO_DIR = r'C:\Users\Yukesh\Downloads\snookervideo'
# # video_path = os.path.join(VIDEO_DIR, '10')  # Change to the specific video you want to process
# # output_video_path = os.path.join(VIDEO_DIR, '{}.mp4')
#
#
# VIDEOS_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
# video_path = os.path.join(VIDEOS_DIR, '10.mp4')
# output_video_path = '{}_outut.mp4'.format(video_path)
#
# model_paths = [
#     os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolofrominternet', 'runs', 'detect', 'train5', 'weights', 'last.pt'),
#     os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolo_8', 'runs', 'detect', 'train10', 'weights', 'last.pt')
# ]
#
# def process_video(video_path, model_paths):
#     # Open the video file
#     video_capture = cv2.VideoCapture(video_path)
#
#     # Get video properties
#     frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
#     num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Create video writer object to save the processed video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
#
#     # Threshold for displaying bounding boxes
#     threshold = 0.5
#
#     for frame_num in range(num_frames):
#         # Read the next frame
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#
#         for model_path in model_paths:
#             # Load the YOLO model
#             model = YOLO(model_path)  # Load the custom model
#
#             # Perform object detection on the frame
#             results = model.predict(frame)
#
#             # Draw bounding boxes on the frame
#             for result in results:
#                 for box in result.boxes:
#                     # Extract box attributes
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     score = box.conf[0]
#                     class_id = box.cls[0]
#
#                     if score > threshold:
#                         # Draw the bounding box
#                         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#
#                         # Define the text and get its width and height
#                         class_name = model.names[int(class_id)].upper() if hasattr(model, 'names') else 'CLASS'
#                         (label_width, label_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#
#                         # Ensure the text fits within the bounding box
#                         if y1 + label_height + baseline < y2:
#                             # Place text inside the bounding box at the top-left corner
#                             cv2.putText(frame, class_name, (int(x1), int(y1 + label_height + baseline)),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
#                         else:
#                             # If not enough space inside, place it above the box
#                             cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
#
#         # Write the processed frame to the output video
#         output_video.write(frame)
#
#     # Release video capture and writer objects
#     video_capture.release()
#     output_video.release()
#
#     print(f"Processed video saved to {output_video_path}")
#
# # Process the specified video with both models
# process_video(video_path, model_paths)


import os
import cv2
import time
from ultralytics import YOLO
from multiprocessing import Process, Queue

# Directory and file paths
VIDEOS_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
video_path = os.path.join(VIDEOS_DIR, 'test-1.mp4')
output_video_path = '{}_outut.mp4'.format(video_path)

model_paths = [
    os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolofrominternet', 'runs', 'detect', 'train5', 'weights', 'last.pt'),
    os.path.join(r'C:\Users\Yukesh\PycharmProjects\yolo_8', 'runs', 'detect', 'train10', 'weights', 'last.pt')
]


def worker(model_path, input_queue, output_queue):
    model = YOLO(model_path)  # Load the YOLO model
    threshold = 0.5

    while True:
        frame = input_queue.get()  # Get frame from input queue
        if frame is None:
            break

        results = model.predict(frame)  # Perform object detection

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                score = box.conf[0]
                class_id = box.cls[0]

                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    class_name = model.names[int(class_id)].upper() if hasattr(model, 'names') else 'CLASS'
                    (label_width, label_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                                            1)

                    if y1 + label_height + baseline < y2:
                        cv2.putText(frame, class_name, (int(x1), int(y1 + label_height + baseline)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        output_queue.put(frame)  # Put processed frame into the output queue


def process_video(video_path, model_paths):
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    input_queues = [Queue() for _ in model_paths]
    output_queues = [Queue() for _ in model_paths]
    processes = []

    for model_path, input_queue, output_queue in zip(model_paths, input_queues, output_queues):
        p = Process(target=worker, args=(model_path, input_queue, output_queue))
        processes.append(p)
        p.start()

    for frame_num in range(num_frames):
        ret, frame = video_capture.read()
        if not ret:
            break

        for input_queue in input_queues:
            input_queue.put(frame.copy())  # Make a copy of the frame for each model

        processed_frames = [output_queue.get() for output_queue in output_queues]

        # Combine frames from all models
        combined_frame = processed_frames[0]  # Initialize with the first model's output
        for frame in processed_frames[1:]:
            combined_frame = cv2.addWeighted(combined_frame, 0.5, frame, 0.5, 0)  # Blend frames

        output_video.write(combined_frame)

    for input_queue in input_queues:
        input_queue.put(None)

    for p in processes:
        p.join()

    video_capture.release()
    output_video.release()

    print(f"Processed video saved to {output_video_path}")


if __name__ == '__main__':
    process_video(video_path, model_paths)
