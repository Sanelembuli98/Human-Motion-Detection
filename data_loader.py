import cv2
import os


class DataLoader:
    def __init__(self, frames_dir, annotations_dir):
        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir

    def load_data(self):
        images = []
        annotations = []

        for video_id in os.listdir(self.frames_dir):
            video_id_dir = os.path.join(self.frames_dir, video_id)

            if not os.path.isdir(video_id_dir):
                continue

            for filename in os.listdir(video_id_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(video_id_dir, filename)
                    annotation_filename = filename.replace('.jpg', '.txt')
                    annotation_path = os.path.join(
                        self.annotations_dir, annotation_filename)

                    # Check if the annotation file exists
                    if not os.path.isfile(annotation_path):
                        print(f"Warning: No annotation found for {filename}")
                        continue

                    image = cv2.imread(image_path)
                    images.append(image)

                    with open(annotation_path, 'r') as file:
                        lines = file.readlines()
                        annotation = [line.strip().split() for line in lines]
                        annotations.append(annotation)

        return images, annotations
