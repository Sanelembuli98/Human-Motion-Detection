import os
import cv2

# Define the directory where you saved the extracted frames
output_dir = 'output_frames'

# List all subdirectories in the output directory
video_dirs = [d for d in os.listdir(
    output_dir) if os.path.isdir(os.path.join(output_dir, d))]

# Iterate through each subdirectory (each corresponding to a video)
for video_dir in video_dirs:
    video_path = os.path.join(output_dir, video_dir)
    frame_files = [f for f in os.listdir(video_path) if f.startswith('frame')]

    # Iterate through each frame file in the subdirectory
    for frame_file in frame_files:
        frame_path = os.path.join(video_path, frame_file)

        # Read the frame using OpenCV
        frame = cv2.imread(frame_path)

        # You can now process or use the 'frame' variable as needed

        # Example: Display the frame (you may need to install a window manager library)
        cv2.imshow('Frame', frame)
        cv2.waitKey(30)  # Delay for 30 milliseconds (adjust as needed)

    cv2.destroyAllWindows()  # Close the display window when done with each subdirectory
