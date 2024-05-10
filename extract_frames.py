import cv2

# Open the video file
video_capture = cv2.VideoCapture("./videos/video_1.mp4")

# Initialize a variable to count frames
frame_count = 0

# Loop through each frame in the video
while True:
    # Read the next frame
    ret, frame = video_capture.read()
    
    # If there are no more frames, break the loop
    if not ret:
        break
    
    # Save the frame as an image
    cv2.imwrite(f"./frames_outdoor/frame_{frame_count}.jpg", frame)
    
    # Increment the frame count
    frame_count += 1

# Release the video capture object
video_capture.release()
