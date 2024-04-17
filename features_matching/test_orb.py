import cv2 as cv
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.transform as transform

def openImgGray(file):
    img = cv.imread(file) # trainImage
    imagegray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    return imagegray

def computeDescriptors(file):
    imagegray=openImgGray(file)
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

f1 = "unwrapped_walgreens_left.jpg"
f2 = "unwrapped_walgreens_right.jpg"
# f1 = "part1.jpg"
# f2 = "part2.jpg"
# f1 = "unrelated1.jpg"
# f2 = "unrelated2.jpg"

img1 = openImgGray(f1)     
img2 = openImgGray(f2)     

kp1, des1 = computeDescriptors(f1)
kp2, des2 = computeDescriptors(f2)

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

thresh = 400

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

    if (x1 < thresh and x2 > img2.shape[0] - thresh):
    # if (x1 < thresh and x2 > img2.shape[0] - thresh and abs(y1 - y2) < 10) or (x1 > img1.shape[0] - thresh and x2 < thresh and abs(y1 - y2) < 10):
        # print("match")
        # print(x1, y1)
        # print(x2, y2)
        edge_matches.append(match)
        src_array.append((x1, y1))
        dst_array.append((x2, y2))
        
print(edge_matches)

src_array = np.array(src_array)
dst_array = np.array(dst_array)

print(src_array)
print(dst_array)

   






# list_kp1 = []
# list_kp2 = []

# # For each match...
# for mat in matches:

#     # Get the matching keypoints for each of the images
#     img1_idx = mat.queryIdx
#     img2_idx = mat.trainIdx

#     # x - columns
#     # y - rows
#     # Get the coordinates
#     (x1, y1) = kp1[img1_idx].pt
#     (x2, y2) = kp2[img2_idx].pt

#     # img1[x1, y1] = 0
#     # Append to each list
#     list_kp1.append((x1, y1))
#     list_kp2.append((x2, y2))

# list_kp1 = np.array(list_kp1)
# list_kp2 = np.array(list_kp2)
src_pts = np.float32([ kp1[m.queryIdx].pt for m in edge_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in edge_matches ]).reshape(-1,1,2)


# M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
M_skimage = transform.estimate_transform('euclidean', src_array, dst_array)
print(M_skimage)


src_array = np.hstack((src_array, np.ones((src_array.shape[0], 1)))).T
dst_array = np.hstack((dst_array, np.ones((dst_array.shape[0], 1)))).T
# print(src_array)
# print(dst_array)

dst_comp = M_skimage.params@src_array
print("diff")
print(dst_comp - dst_array)
# print(M)




# exit()

# computed_dst = M@src_array
# print(computed_dst)


# exit()


# M, mask = cv.findHomography(list_kp1, list_kp2, cv.RANSAC,5.0)
final_img = np.zeros(2*np.array(img2.shape))
final_img[0:img2.shape[0], 0:img2.shape[1]] = img2 


img_mod = np.zeros_like(img2)

img_mod = cv.warpPerspective(img1, M_skimage.params, 2*np.array(img_mod.shape[::-1]))
# img_mod = transform.warp(img2, M_skimage.inverse)

# fig, axs = plt.subplots(1, 2)

# print(final_img)

final_shape = final_img.shape

# for x in range(final_shape[0]):
#     for y in range(final_shape[1]):
#         final_img[x][y] += img_mod[x][y]
#         if final_img[x][y] != 0 and img_mod[x][y] != 0:
#             final_img[x][y] /= 2

# final_img += img_mod

nonzero_mask = (img_mod != 0) & (final_img !=0)
final_img += img_mod
final_img[nonzero_mask] /= 2

x, y = np.nonzero(final_img)
xl,xr = x.min(),x.max()
yl,yr = y.min(),y.max()


# final_img = final_img[xl:xr+1, yl:yr+1]
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


fig, axs = plt.subplots(2)
fig.set_size_inches(15, 12)
# axs[0].imshow(imageWithKeypoints("part1.jpg"))  
# axs[1].imshow(imageWithKeypoints("part2.jpg"))  
axs[0].imshow(img3)
axs[0].axis(False)
axs[0].set_title("KeyPoints matching between two images")
axs[1].imshow(final_img, cmap="gray")
axs[1].axis(False)
axs[1].set_title("Stitched image")
plt.tight_layout()
# plt.savefig("test_stiching.jpg", dpi=300)

plt.show() 

