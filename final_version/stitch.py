import cv2 as cv
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.transform as transform
import sys
from image_cutting_unwrapping import unwrap_and_cut_img
from skimage.color import rgb2gray, gray2rgb
from PIL import Image as im 
import os

def openImgGray(file):
    img = cv.imread(file) # trainImage
    imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    return imagegray

def computeDescriptors(img, use_img = False):
    if use_img == False:
        imagegray=openImgGray(img)
    else: 
        imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    # # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(imagegray,None)
    return kp, des


def imageWithKeypoints(file):
    img = cv.imread(file) # trainImage
    imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    orb = cv.ORB_create()
    # # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(imagegray,None)
    output_image = cv.drawKeypoints(imagegray, kp, 0, (0, 255, 0), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    return output_image

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
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        if abs(y1- y2) <20 and y1 > 0.35*height and y1 < 0.65*height: # we restrict features not aligned horizontally
            edge_matches.append(match)
            src_array.append((x1, y1))
            dst_array.append((x2, y2))

    src_array = np.array(src_array)
    dst_array = np.array(dst_array)
    return src_array, dst_array, edge_matches




def stitch(filename):

    img1_full, img2_full = unwrap_and_cut_img(f"{filename}", 190) # here img1 are open cv images so coordinates are reversed
    # h, w = img1_full.shape
    
    # plt.imshow(img2_full)
    # plt.show()
    
    width = int(img1_full.shape[1]*(190-185)/100)
    height = img1_full.shape[0]
  
    img1_1 = img1_full[:, 0:width, :]
    img2_1 = img2_full[:, -width:, :]
    
    kp1_1, des1_1 = computeDescriptors(img1_1, use_img=True)
    kp2_1, des2_1 = computeDescriptors(img2_1, use_img=True)
   
    src_array_1, dst_array_1, edges_match_1 = computeMatches(des1_1, des2_1, kp1_1, kp2_1, height)
    
    src_array = src_array_1
    dst_array = dst_array_1
   

    M_skimage = transform.estimate_transform('euclidean', src_array, dst_array)
    tr_matrix = M_skimage.params
    # # print(f"trans - {tr_matrix[0, 2]}")
    # tr_matrix = np.load("tr_matrix.npy")
    
    overlapping_width = int(width - tr_matrix[0, 2]) + 1
    tr_matrix[0, 0] = 1
    tr_matrix[0, 1] = 0
    tr_matrix[1, 0] = 0
    tr_matrix[1, 1] = 1
    tr_matrix[1, 2] = 1
    # np.save("tr_matrix.npy", tr_matrix)
    tr_matrix[0, 2] = (img1_full.shape[1] - width + tr_matrix[0, 2])
    # print(tr_matrix)
    
 
    
    # print(tr_matrix)

    # overlapping_width = int(np.ceil(img1.shape[1] - M_skimage.params[0, 2]))
    # print("overlapping width: ", overlapping_width)


    src_array = np.hstack((src_array, np.ones((src_array.shape[0], 1)))).T
    dst_array = np.hstack((dst_array, np.ones((dst_array.shape[0], 1)))).T

   
    img2_full[:, 0:overlapping_width, :] = 0
  
    shape = [2*img2_full.shape[0], 2*img2_full.shape[1], 3]
    final_img = np.zeros(shape)
    
    final_img[0:img2_full.shape[0], 0:img2_full.shape[1], :] = img2_full



    img_mod = np.zeros_like(img2_full)

    img1_full = img1_full.astype(np.float32)
    #blending left part of the image on the right
    for i in range(overlapping_width):
        img1_full[:, i, :]*=(i/overlapping_width)
    img1_full = img1_full.astype(np.uint8)

    
    img_mod = cv.warpPerspective(img1_full, tr_matrix, [2*img2_full.shape[1], 2*img2_full.shape[0]] )

    final_img += img_mod
    final_img = final_img.astype(np.float64)
    for i in range(overlapping_width):
        final_img[:, int(tr_matrix[0, 2])+ i, :] /= (1 + i/overlapping_width)
    final_img = final_img.astype(np.uint8)



    x, y, z = np.nonzero(final_img)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()


    final_img = final_img[xl:xr+1, yl:yr+1, :]

    rgb = np.array(final_img)
    # rgb[:, :, 0] = final_img[:, :, 2]
    # rgb[:, :, 2] = final_img[:, :, 0]

    rgb = rgb.astype(np.uint8)
    name, ext = os.path.splitext(os.path.basename(filename))
    if not os.path.exists("./stitched"):
        os.mkdir("stitched")


    im.fromarray(rgb).save(f"./stitched/{name}_stitched.jpg")


arg1 = sys.argv[1]

for i in range(1, len(sys.argv)):
    stitch(sys.argv[i])
    


