import cv2
import numpy as np
import os
import data_loader as dl
import __generic_weights as weights
import losses as loss


def __ini__():
    print("Execution starts...")


# Define the parent directory where all the video subdirectories are located
parent_dir = 'output_frames/'
labels_dir = 'annotations/'

# Get a list of video subdirectories
video_subdirs = [video_dir for video_dir in os.listdir(
    parent_dir) if os.path.isdir(os.path.join(parent_dir, video_dir))]

# Initialize lists to store image data and labels
images = []
labels = []

# Loop through each video subdirectory
for video_dir in video_subdirs:
    # Define the full path to the video directory
    video_dir_path = os.path.join(parent_dir, video_dir)

    # Get a list of image file names within the video directory (assuming they have .jpg extension)
    image_file_names = os.listdir(video_dir_path)
    image_file_names = [
        file for file in image_file_names if file.endswith('.jpg')]

    # Loop through each image file in the video directory
    for image_file_name in image_file_names:
        # Load image
        image_path = os.path.join(video_dir_path, image_file_name)
        image = cv2.imread(image_path)

        # Load corresponding annotation file (if needed)
        annotation_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        annotation_path = os.path.join("annotations", annotation_file_name)
        images.append(image)
        labels.append(...)

# Convert image and label lists to NumPy arrays (if needed)
# images = np.array(images)
# labels = np.array(labels)

# print(images, labels)

input_shape = (3, 224, 224)
kernel_size = 3
depth = 1

# Model Initialization
# model = cnn.Cnn(input_shape, kernel_size, depth)

# Load data using the data loader
data_loader = dl.DataLoader(
    annotations_dir=labels_dir, frames_dir=parent_dir)

images, annotations = data_loader.load_data()

# Training Loop

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    total_loss = 0

    for image, annotation in zip(images, annotations):
        # Check if annotation is a list and has at least one element
        if isinstance(annotation, list) and len(annotation) > 0:
            # Assuming the file path is the first element in the list
            annotation_path = annotation[0]
            try:
                with open(annotation_path, 'r') as file:
                    annotation_data = file.read()
                # parsed_annotation = parse_annotation_data(annotation_data)
                print(f"Processing Annotation: {annotation_path}")
            except FileNotFoundError:
                print(f"")
            except Exception as e:
                print(f"")
        else:
            print("Invalid annotation:", annotation)

# Load the pre-trained model
model = model = weights.Test()

# Define the class labels
class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'human', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor']

# Open a video capture object (you can use 0 for the default camera or specify a video file)
cap = cv2.VideoCapture(0)  # Replace 'video.mp4' with your video file

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Prepare the frame for detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the model
    model.setInput(blob)

    # Perform inference and get the results
    detections = model.forward()

    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above a certain threshold (e.g., 0.5), consider it a human detection
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = class_labels[class_id]

            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype('int')

            # Draw the bounding box and label
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


def preprocess_image(image):
    preprocessed_image = perform_preprocessing_steps(image)
    return preprocessed_image


def perform_preprocessing_steps(image):
    # Resize the image to a specific size (e.g., 224x224 pixels)
    resized_image = cv2.resize(image, (224, 224))

    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    return normalized_image


def preprocess_annotations(annotation_paths):
    parsed_annotations = []
    for annotation_path in annotation_paths:
        try:
            with open(annotation_path, 'r') as file:
                annotation_data = file.read()
            parsed_annotation = parse_annotation_data(annotation_data)
            parsed_annotations.append(parsed_annotation)
        except FileNotFoundError:
            print(f"File not found: {annotation_path}")
        except Exception as e:
            print(
                f"An error occurred while processing {annotation_path}: {str(e)}")
    return parsed_annotations


def parse_annotation_data(annotation_data):
    # Split the data and extract bounding box coordinates (modify this based on your format)
    data_parts = annotation_data.split(',')
    x_min, y_min, x_max, y_max = map(int, data_parts[:4])
    class_label = data_parts[4].strip()

    # Return a structured annotation
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'class_label': class_label
    }
