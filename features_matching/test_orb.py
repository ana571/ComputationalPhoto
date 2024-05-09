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

f1 = "left.jpg"
f2 = "right.jpg"
# f1 = "./imgs/unwrapped_walgreens_left.jpg"
# f2 = "./imgs/unwrapped_walgreens_right.jpg"
# f1 = "part1.jpg"
# f2 = "part2.jpg"
# f1 = "unrelated1.jpg"
# f2 = "unrelated2.jpg"

width = 150

img1_full, img2_full = unwrap_and_cut_img("./test.jpg", f1, f2, 190)
img1 = img1_full
img2 = img2_full
# img1 = img1_full[:, 0:width, :]
# img2 = img2_full[:, -width:, :]

# fig, axs = plt.subplots(1, 2)

# axs[0].imshow(img1[:,:,::-1])
# axs[0].axis(False)
# axs[1].imshow(img2[:,:,::-1])
# axs[1].axis(False)

# plt.tight_layout()
# plt.show()


# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
# exit()
# img1 = openImgGray(f1)
# img2 = openImgGray(f2)
# img1 = rgb2gray(img1.T)
# img2 = rgb2gray(img2)
# plt.imshow(img1, cmap="gray")
# plt.show()


# exit(0)

kp1, des1 = computeDescriptors(f1, use_img=False)
kp2, des2 = computeDescriptors(f2, use_img=False)
# kp1, des1 = computeDescriptors(img1, use_img=True)
# kp2, des2 = computeDescriptors(img2, use_img=True)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
 
# Match descriptors.
matches = bf.match(des1,des2)
 
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# print(matches[10].distance)
 
n_match = 20
matches = matches[:n_match]
edge_matches = []
print("dimension 1")
print(img1.shape)
print("dimension 2")
print(img2.shape)

#TODO change as percentage
thresh = 100

src_array = []
dst_array = []




for match in matches:
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    print("distance")
    print(match.distance)

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    if ((x1 < thresh and x2 > img2.shape[0] - thresh)):
        # print("distance")
    # if x1 > img1.shape[0] - thresh and x2 < thresh :
        # print("matched ", match.distance)
        # print("match")
        # print(x1, y1)
        # print(x2, y2)
    # if match.distance < 25:
        edge_matches.append(match)
        src_array.append((x1, y1))
        dst_array.append((x2, y2))
    
# print(edge_matches)

src_array = np.array(src_array)
dst_array = np.array(dst_array)

print(src_array)
print(dst_array)

   


src_pts = np.float32([ kp1[m.queryIdx].pt for m in edge_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in edge_matches ]).reshape(-1,1,2)


# M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
M_skimage = transform.estimate_transform('euclidean', src_array, dst_array)
print(M_skimage)
tr_matrix = M_skimage.params
tr_matrix[0, 0] = 1
tr_matrix[0, 1] = 0
tr_matrix[1, 0] = 0
tr_matrix[1, 1] = 1
tr_matrix[1, 2] = 1
# tr_matrix[0, 2] = (img1_full.shape[1] - width + 1.5*tr_matrix[0, 2])

overlapping_width = int(np.ceil(img1.shape[1] - M_skimage.params[0, 2]))
print("overlapping width: ", overlapping_width)


src_array = np.hstack((src_array, np.ones((src_array.shape[0], 1)))).T
dst_array = np.hstack((dst_array, np.ones((dst_array.shape[0], 1)))).T

# dst_comp = M_skimage.params@src_array
# print("diff")
# print(dst_comp - dst_array)

# M, mask = cv.findHomography(list_kp1, list_kp2, cv.RANSAC,5.0)

shape = [2*img2_full.shape[0], 2*img2_full.shape[1], 3]
final_img = np.zeros(shape)
final_img[0:img2_full.shape[0], 0:img2_full.shape[1], :] = img2_full



img_mod = np.zeros_like(img2_full)

img1_full = img1_full.astype(np.float32)
for i in range(overlapping_width):
    img1_full[:, i, :]*=(i/overlapping_width)
img1_full = img1_full.astype(np.uint8)
img_mod = cv.warpPerspective(img1_full, tr_matrix, [2*img2_full.shape[1], 2*img2_full.shape[0]] )


    
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

final_img[:, 0:overlapping_width, :] =0


x, y, z = np.nonzero(final_img)
xl,xr = x.min(),x.max()
yl,yr = y.min(),y.max()


final_img = final_img[xl:xr+1, yl:yr+1, :]
print(final_img.shape)

rgb = np.array(final_img)
rgb[:, :, 0] = final_img[:, :, 2]
rgb[:, :, 2] = final_img[:, :, 0]

rgb = rgb.astype(np.uint8)

im.fromarray(rgb).save("./test/stitched.jpg")

# plt.imshow(final_img, cmap="gray")
# plt.show()
# exit()
# axs[0].imshow(final_img, cmap="gray")  
# axs[1].imshow(img_mod, cmap="gray")  
# axs[2].imshow(nonzero_mask, cmap="gray")  
# plt.show() 


# for i in range(n_match):
#     trainIdx = matches[i].trainIdx
#     print(kp1[trainIdx].pt)
# # # # Draw first 10 matches.
img3 = cv.drawMatches(img1, kp1,img2,kp2,edge_matches,None, matchesThickness=2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.axis(False)
# plt.tight_layout()

# plt.imshow(img1, cmap="gray")
# plt.show()
# plt.savefig("not_matching.jpg", dpi=300)


fig, axs = plt.subplots(1, 2)
fig.set_size_inches(15, 12)
# axs[0].imshow(imageWithKeypoints("part1.jpg"))  
# axs[1].imshow(imageWithKeypoints("part2.jpg"))  
axs[0].imshow(img3)
axs[0].axis(False)
axs[0].set_title("KeyPoints matching between two images")
axs[1].imshow(rgb)
axs[1].axis(False)
axs[1].set_title("Stitched image")
plt.tight_layout()
# plt.savefig("test_stiching.jpg", dpi=300)

plt.show() 

