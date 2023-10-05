import json
import os
import cv2
from moviepy.editor import VideoFileClip
from pytube import YouTube

# Define the path to your JSON file containing video metadata
json_file = 'ds_kinetics_700_2020/train.json'

# Define the directory where you'll save extracted frames
output_dir = 'output_frames'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Function to extract frames from a video file
def extract_frames(video_path, output_folder):
    clip = VideoFileClip(video_path)
    frame_count = 0

    for frame in clip.iter_frames(fps=25):  # Adjust the frame rate as needed
        frame_filename = f"frame{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)

        # Save the frame as an image
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_count += 1

    return frame_count

# Load video URLs from the JSON file and download/extract frames


with open(json_file, 'r') as f:
    video_data = json.load(f)

for video_id, video_info in video_data.items():
    youtube_url = video_info['url']
    video_output_dir = os.path.join(output_dir, video_id)

    os.makedirs(video_output_dir, exist_ok=True)

    # Download the YouTube video
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension="mp4").first()
        stream.download(output_path=video_output_dir)
    except Exception as e:
        print(f"Error downloading video {video_id}: {str(e)}")
        continue

    # Extract frames from the downloaded video
    video_path = os.path.join(video_output_dir, stream.default_filename)
    frame_count = extract_frames(video_path, video_output_dir)

    print(f"Extracted {frame_count} frames from video {video_id}")
    # break
print("Frames extracted successfully.")
