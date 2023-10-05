import cv2
import numpy as np
import os

# Load the pre-trained MobileNet-SSD model for object detection
net = cv2.dnn.readNetFromCaffe(
    '__ssd__helper__/MobileNetSSD_deploy.caffemodel',
    '__ssd__helper__/MobileNetSSD_deploy.prototxt'
)

# Define the path to the directory containing your frames
frames_dir = 'output_frames'

# Define the directory where you'll save annotation files
annotations_dir = 'annotations'

# Create the annotations directory if it doesn't exist
os.makedirs(annotations_dir, exist_ok=True)

# Process each subdirectory (video ID directory) in frames_dir
for video_id in os.listdir(frames_dir):
    video_dir = os.path.join(frames_dir, video_id)

    if os.path.isdir(video_dir):
        # Process each frame in the subdirectory
        for frame_filename in os.listdir(video_dir):
            if frame_filename.endswith('.jpg'):
                frame_path = os.path.join(video_dir, frame_filename)
                frame = cv2.imread(frame_path)

                # Prepare the frame for object detection
                blob = cv2.dnn.blobFromImage(
                    frame, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # Iterate over the detected objects
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    # Filter out low-confidence detections
                    if confidence > 0.5:
                        class_id = int(detections[0, 0, i, 1])

                        # Check if the detected object is a human (class_id 15)
                        if class_id == 15:
                            # Get the bounding box coordinates
                            box = detections[0, 0, i, 3:7] * np.array(
                                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                            (startX, startY, endX, endY) = box.astype('int')

                            # Create an annotation file with the bounding box coordinates
                            annotation_filename = os.path.splitext(frame_filename)[
                                0] + '.txt'
                            annotation_path = os.path.join(
                                annotations_dir, annotation_filename)
                            with open(annotation_path, 'w') as annotation_file:
                                annotation_file.write(
                                    f'0 {startX} {startY} {endX} {endY}')

print("Automated labeling completed.")
