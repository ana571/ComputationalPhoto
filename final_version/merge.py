import cv2
import os
from tqdm import tqdm
import re
def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else -1

def create_video(image_folder, output_video, fps=30):
    # Get the list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=extract_frame_number)  # Ensure the images are sorted in the correct order
    print(images)

    if not images:
        print("No images found in the folder")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(image_folder, output_video), fourcc, fps, (width, height))

    for image in tqdm(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer object
    video.release()

    print(f"Video saved as {output_video}")

# Example usage
image_folder = 'merge-cv'  # Replace with the path to your image folder
output_video = 'stitched.mp4'  # The name of the output video file
fps = 30  # Frames per second

create_video(image_folder, output_video, fps)