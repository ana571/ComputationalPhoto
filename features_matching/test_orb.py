import cv2 as cv
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.transform as transform
import sys
from image_cutting_unwrapping import unwrap_and_cut_img
from skimage.color import rgb2gray, gray2rgb
from PIL import Image as im 

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
    # print(matches[10].distance)
    
    n_match = 50
    matches = matches[:n_match]
    edge_matches = []
    # print("dimension 1")
    # print(img1.shape)
    # print("dimension 2")
    # print(img2.shape)

    src_array = []
    dst_array = []




    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        # print("distance")
        # print(match.distance)

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # if ((x1 < thresh and x2 > img2.shape[0] - thresh)):
            # print("distance")
        # if x1 > img1.shape[0] - thresh and x2 < thresh :
        # print("matched ", match.distance)
            # print("match")
            # print(x1, y1)
            # print(x2, y2)
        # if match.distance < 35: #TODO check rule for distance
        if abs(y1- y2) <20 and y1 > 0.35*height and y1 < 0.65*height: # we restrict features not aligned horizontally
            edge_matches.append(match)
            src_array.append((x1, y1))
            dst_array.append((x2, y2))
            # print(f"matched: {match.distance}")

    # print(edge_matches)

    src_array = np.array(src_array)
    dst_array = np.array(dst_array)
    return src_array, dst_array, edge_matches




def stitch(file_name, debug=False):

    f1 = "left.jpg"
    f2 = "right.jpg"
    # f1 = "./imgs/unwrapped_walgreens_left.jpg"
    # f2 = "./imgs/unwrapped_walgreens_right.jpg"
    # f1 = "part1.jpg"
    # f2 = "part2.jpg"
    # f1 = "unrelated1.jpg"
    # f2 = "unrelated2.jpg"


    img1_full, img2_full = unwrap_and_cut_img(f"../frames_outdoor/{file_name}", f1, f2, 190) # here img1 are open cv images so coordinates are reversed
    # h, w = img1_full.shape
    width = int(img1_full.shape[1]*(190-185)/100)
    height = img1_full.shape[0]
    # h, w = img1_full.size
    # img1 = img1_full
    # img2 = img2_full
    img1_1 = img1_full[:, 0:width, :]
    img2_1 = img2_full[:, -width:, :]
    
    img1_2 = img2_full[:, 0:width, :]
    img2_2 = img1_full[:, -width:, :]
    
    # if debug:
    #     fig, axs = plt.subplots(1, 2)

    #     axs[0].imshow(img1[:,:,::-1])
    #     axs[0].axis(False)
    #     axs[1].imshow(img2[:,:,::-1])
    #     axs[1].axis(False)

    #     plt.tight_layout()
    #     plt.show()





    kp1_1, des1_1 = computeDescriptors(img1_1, use_img=True)
    kp2_1, des2_1 = computeDescriptors(img2_1, use_img=True)
    
    kp1_2, des1_2 = computeDescriptors(img1_2, use_img=True)
    kp2_2, des2_2 = computeDescriptors(img2_2, use_img=True)
    # kp1, des1 = computeDescriptors(img1, use_img=True)
    # kp2, des2 = computeDescriptors(img2, use_img=True)
    
    src_array_1, dst_array_1, edges_match_1 = computeMatches(des1_1, des2_1, kp1_1, kp2_1, height)
    # print(f"match nb1 : {len(edges_match_1)}")
    src_array_2, dst_array_2, edges_match_2 = computeMatches(des1_2, des2_2, kp1_2, kp2_2, height)
    # print(f"match nb2 : {len(edges_match_2)}")
    
    direction = 1    
    src_array = src_array_1
    dst_array = dst_array_1
    # if (len(edges_match_2) > len(edges_match_1)):
    #     direction = 2
    #     src_array = src_array_2
    #     dst_array = dst_array_2
    #     temp1 = img1_full.copy()
    #     img1_full = img2_full.copy()
    #     img2_full = temp1


        
        



    # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    M_skimage = transform.estimate_transform('euclidean', src_array, dst_array)
    tr_matrix = M_skimage.params
    # # print(f"trans - {tr_matrix[0, 2]}")
    # tr_matrix = np.load("tr_matrix.npy")
    print(tr_matrix)
    
    overlapping_width = int(width - tr_matrix[0, 2]) + 1
    tr_matrix[0, 0] = 1
    tr_matrix[0, 1] = 0
    tr_matrix[1, 0] = 0
    tr_matrix[1, 1] = 1
    tr_matrix[1, 2] = 1
    # np.save("tr_matrix.npy", tr_matrix)
    tr_matrix[0, 2] = (img1_full.shape[1] - width + tr_matrix[0, 2])
    # print(tr_matrix)
    
 
    
    print(tr_matrix)

    # overlapping_width = int(np.ceil(img1.shape[1] - M_skimage.params[0, 2]))
    # print("overlapping width: ", overlapping_width)


    src_array = np.hstack((src_array, np.ones((src_array.shape[0], 1)))).T
    dst_array = np.hstack((dst_array, np.ones((dst_array.shape[0], 1)))).T

    # dst_comp = M_skimage.params@src_array
    # print("diff")
    # print(dst_comp - dst_array)

    # M, mask = cv.findHomography(list_kp1, list_kp2, cv.RANSAC,5.0)
    
    #cropping the overlapping part of the left
    # img2_full[:, 0:overlapping_width//2, :] = 0
    
    overlap_left = np.array(img2_full[:, 0:overlapping_width, :]).astype(np.float32)
    overlap_right = np.array(img1_full[:, -overlapping_width:, :]).astype(np.float32)
    
    
    # for i in range (overlapping_width):
    #     overlap_left[:, i, :] *= (1-i/overlapping_width)
    #     overlap_right[:, i, :] *= (i/overlapping_width)
    
    overlap_blend = ((overlap_left/2+overlap_right/2)).astype(np.uint8)
    # ob_copy = overlap_blend.copy()
    # overlap_blend[:, :, 0] = overlap_blend[:, :, 2]
    # overlap_blend[:, :, 2] = ob_copy[:, :, 0]
    
    
    # plt.imshow(overlap_blend)
    # plt.show()
    
    # overlap_blend = np.zeros_like(overlap_left)
    # overlap_blend = ((overlap_left + overlap_right)/2).astype(np.uint8)
    
    
    # print(overlap_blend.shape)
    # print(overlap_blend.max())

    # # plt.imshow(overlap_blend)
    # # plt.show()
    img2_full[:, 0:overlapping_width, :] = 0
    # img1_full[:, -overlapping_width:, :] = overlap_blend
    # plt.imshow(overlap_left)
    # plt.show()
    # plt.imshow(overlap_right)
    # plt.show()
    # mid = overlapping_width//2
    # for i in range(overlapping_width):
    #     overlap_blend[:, i, :] = 1-i/overlapping_width
        
    
    
    # img2_full = img2_full.astype(np.float32)
    # #blending left part of the image on the right
    # for i in range(overlapping_width):
    #     img2_full[:, i, :]*=(1-i/overlapping_width)
    # img2_full = img2_full.astype(np.uint8)
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



    if debug:
        plt.imshow(img_mod)
        plt.show()
    # img_mod = transform.warp(img2, M_skimage.inverse)


    # fig, axs = plt.subplots(1, 2)

    # print(final_img)

    final_shape = final_img.shape

    nonzero_mask = (img_mod != 0) & (final_img !=0)

    # img_mod_mask = (img_mod != 0)[:, :, 0].astype(np.uint8)

    # # r = nonzero_mask[:, :, 0].astype(np.uint8)
    # plt.imshow(img_mod_mask)
    # plt.show()
    # print(np.argwhere(r ==1))
    final_img += img_mod
    # final_img[nonzero_mask] /= 2
    final_img = final_img.astype(np.float64)
    for i in range(overlapping_width):
        final_img[:, int(tr_matrix[0, 2])+ i, :] /= (1 + i/overlapping_width)
    final_img = final_img.astype(np.uint8)

    # final_img[:, 0:overlapping_width, :] =0


    x, y, z = np.nonzero(final_img)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()


    final_img = final_img[xl:xr+1, yl:yr+1, :]
    # print(final_img.shape)

    rgb = np.array(final_img)
    rgb[:, :, 0] = final_img[:, :, 2]
    rgb[:, :, 2] = final_img[:, :, 0]

    rgb = rgb.astype(np.uint8)
    crop_width = 50
    if direction == 1:
        # rgb = np.roll(rgb, overlapping_width//2, axis=1)
        rgb = np.roll(rgb, rgb.shape[1]//2, axis=1)
        # middle = rgb.shape[1]//2
        # blur_overlap = cv.blur(rgb[:, middle-50:middle+50, :], (5, 5))
        # rgb[:, middle-50:middle+50, :] = blur_overlap
        # rgb = np.roll(rgb, rgb.shape[1]//2, axis=1)
        
        pass
        # rgb = rgb[:, crop_width:-crop_width, :]


    im.fromarray(rgb).save(f"../frames_stitched_outdoor/{file_name}")





    if debug:
        img3 = cv.drawMatches(img1, kp1,img2,kp2,edge_matches,None, matchesThickness=2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(15, 12)
        axs[0].imshow(img3)
        axs[0].axis(False)
        axs[0].set_title("KeyPoints matching between two images")
        axs[1].imshow(rgb)
        axs[1].axis(False)
        axs[1].set_title("Stitched image")
        plt.tight_layout()

        plt.show() 


# start = 127

for i in range (100, 101):

    print(f"stitching {i}")
    stitch(f"frame_{i}.jpg", False)
        
