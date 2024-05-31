import cv2 as cv
import numpy as np
import skimage.transform as transform
import sys
from image_cutting_unwrapping import unwrap_and_cut_img
from PIL import Image as im 
import os
from tqdm import tqdm
import re

# used to sort the frames by their number.
def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else -1

# create a video from a folder of frames.
def create_video(image_folder, output_video, fps=30):
    # Get the list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=extract_frame_number)  # Ensure the images are sorted in the correct order

    if not images:
        print("No images found in the folder")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    path = os.path.join(image_folder, output_video)
    video = cv.VideoWriter(path, fourcc, fps, (width, height))

    for image in tqdm(images):
        image_path = os.path.join(image_folder, image)
        frame = cv.imread(image_path)
        frame = cv.resize(frame, (width, height), cv.INTER_CUBIC)
        video.write(frame)

    # Release the video writer object
    video.release()

    print(f"Video saved as {path}")

# delete a directory. Used to delete the temporary directories.
def remove_non_empty_directory(directory_path):
    if not os.path.exists(directory_path): return
    # Walk the directory tree
    for root, dirs, files in os.walk(directory_path, topdown=False):
        # Remove all files
        for file in files:
            os.remove(os.path.join(root, file))
        # Remove all directories
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    # Finally, remove the main directory
    os.rmdir(directory_path)

# open image as grayscale
def openImgGray(file):
    img = cv.imread(file) # trainImage
    imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    return imagegray

# compute the keypoints and descriptors for an image
def computeDescriptors(img, use_img = False):
    if use_img == False:
        imagegray=openImgGray(img)
    else: 
        imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    # # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(imagegray,None)
    return kp, des

# get the coordinates of the matching points
def computeMatches(des1, des2, kp1, kp2, height):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    n_match = 50
    matches = matches[:n_match]
    edge_matches = []
   
    src_array = []
    dst_array = []

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
      
        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # we only select features aligned horizontally and located around the center of the image
        if abs(y1- y2) <20 and y1 > 0.35*height and y1 < 0.65*height: 
            edge_matches.append(match)
            src_array.append((x1, y1))
            dst_array.append((x2, y2))

    src_array = np.array(src_array)
    dst_array = np.array(dst_array)
    return src_array, dst_array, edge_matches

# extract the frames from a video file
def extract_frames(frames_dir, video_path):
    video_capture = cv.VideoCapture(video_path)

    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        # If there are no more frames, break the loop
        if not ret:
            break
        
        # Save the frame as an image
        cv.imwrite(f"./{frames_dir}/frame_{frame_count}.jpg", frame)
        
        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()


# main function to stitch an image
def stitch(filename, video = False, precomputed = False):
    # here img1_full, img2_full are open cv images so coordinates are reversed
    img1_full, img2_full = unwrap_and_cut_img(f"{filename}", 190) 

    #coarse guess of the overlapping width
    width = int(img1_full.shape[1]*(190-185)/100)

    if precomputed:
        tr_matrix = np.load("tr_matrix.npy")
    else:
        height = img1_full.shape[0]

        # select sides of the image where we will do features matching
        img1_1 = img1_full[:, 0:width, :]
        img2_1 = img2_full[:, -width:, :]
        
        # compute keypoints and descriptors
        kp1_1, des1_1 = computeDescriptors(img1_1, use_img=True)
        kp2_1, des2_1 = computeDescriptors(img2_1, use_img=True)
    
        # retrieve matching coordinates
        src_array, dst_array, edges_match_1 = computeMatches(des1_1, des2_1, kp1_1, kp2_1, height)
        
        if len(src_array) < 2 or len(dst_array) < 2: return

        M_skimage = transform.estimate_transform('euclidean', src_array, dst_array)
        tr_matrix = M_skimage.params
    
    # compute the overlapping width
    overlapping_width = int(width - tr_matrix[0, 2]) + 1

    # only keep the x-translation
    tr_matrix[0, 0] = 1
    tr_matrix[0, 1] = 0
    tr_matrix[1, 0] = 0
    tr_matrix[1, 1] = 1
    tr_matrix[1, 2] = 1

    # update the x-translation value
    tr_matrix[0, 2] = (img1_full.shape[1] - width + tr_matrix[0, 2])

    # remove the other overlapping part
    img2_full[:, 0:overlapping_width, :] = 0
  
    # shape of the final image
    shape = [2*img2_full.shape[0], 2*img2_full.shape[1], 3]
    final_img = np.zeros(shape)
    
    # we put the second image to the left of the final image
    final_img[0:img2_full.shape[0], 0:img2_full.shape[1], :] = img2_full

    img_mod = np.zeros_like(img2_full)

    img1_full = img1_full.astype(np.float32)
    #blending left part of the image on the right
    for i in range(overlapping_width):
        img1_full[:, i, :]*=(i/overlapping_width)
    img1_full = img1_full.astype(np.uint8)

    # transform the first image with our transformation matrix
    img_mod = cv.warpPerspective(img1_full, tr_matrix, [2*img2_full.shape[1], 2*img2_full.shape[0]] )

    # adding both image together
    final_img += img_mod

    # blending right part
    final_img = final_img.astype(np.float64)
    for i in range(overlapping_width):
        final_img[:, int(tr_matrix[0, 2])+ i, :] /= (1 + i/overlapping_width)
    final_img = final_img.astype(np.uint8)

    # crop the zero part of the image
    x, y, z = np.nonzero(final_img)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    final_img = final_img[xl:xr+1, yl:yr+1, :]
    
    # convert to uint8 array before saving
    rgb = np.array(final_img)
    rgb = rgb.astype(np.uint8)
    name, ext = os.path.splitext(os.path.basename(filename))
    if video:
        if not os.path.exists("./video"):
            os.mkdir("video")
        im.fromarray(rgb).save(f"./video/{name}.jpg")
        
    else:
        if not os.path.exists("./stitched"):
            os.mkdir("stitched")
        im.fromarray(rgb).save(f"./stitched/{name}_stitched.jpg")



arg1 = sys.argv[1]
arg2 = sys.argv[1]

precomputed = False

# check if we want to load our matrix
if arg1 == "-p" or arg1 == "--precomputed":
    precomputed = True
    arg2 = sys.argv[2]

# check if we stitch a video
if arg2 != "-v" and arg2 != "--video":
    start = 1
    if precomputed: start = 2
    for i in range(start, len(sys.argv)):
        if os.path.isdir(sys.argv[i]): continue
        stitch(sys.argv[i], precomputed=precomputed)
        
else:
    video_dir = "video"
    remove_non_empty_directory(video_dir)
    start = 2
    if precomputed: start = 3
    video_path = sys.argv[start]
    frames_dir = "frames"
    remove_non_empty_directory(frames_dir)
    os.mkdir(frames_dir)
    
    print("extracting frames")
    extract_frames(frames_dir, video_path)
        
    print("stitching frames")

    for file in tqdm(os.listdir(frames_dir)):
        path = os.path.join(frames_dir, file)        
        stitch(path, video=True, precomputed=precomputed)
        os.remove(path)
    
    print("mounting video")
    create_video(video_dir, "stitched_video.mp4", 30)
    
    remove_non_empty_directory(frames_dir)
    for video_file in os.listdir(video_dir):
        path = os.path.join(video_dir, video_file)
        if not path.endswith(".mp4"):
            os.remove(path)
        
    
    


