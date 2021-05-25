import cv2       # OpenCV
import numpy as np
# 1. Load the original image
#img = cv2.imread('house.jpg')
img = cv2.imread('images//lucca.jpg')
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2. Create BRISK algorithm
# OpenCV default threshold = 30, octaves = 3
# Using 4 octaves as cited as typical value by the original paper by Leutenegger et al.
# Using 70 as detection threshold similar to real-world example of this paper
brisk = cv2.BRISK_create(70, 4)
# 3. Combined call to let the BRISK implementation detect keypoints
# as well as calculate the descriptors, based on the grayscale image.
# These are returned in two arrays.
(kps, descs) = brisk.detectAndCompute(gray, None)
# 4. Print the number of keypoints and descriptors found
print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
#print(kps)
#print("diferentes:", len(np.unique(kps)))

# coords=[x+1j*y for (x,y) in descs] # using complex; a way for grouping
# uniques,counts=np.unique(coords, axis = 0, return_counts=True)
# res=[[x.real,x.imag] for x in uniques[counts==1] ] # ungroup
arr, uniq_cnt = np.unique(descs, axis=0, return_counts=True)
uniq_arr = arr[uniq_cnt==1]

#print(uniq_arr)
print("a", len(uniq_arr))
# To verify: how many bits are contained in a feature descriptor?
# Should be 64 * 8 = 512 bits according to the algorithm paper.
print(len(descs[1]) * 8)
# 5. Use the generic drawKeypoints method from OpenCV to draw the 
# calculated keypoints into the original image.
# The flag for rich keypoints also draws circles to indicate
# direction and scale of the keypoints.

#imgBrisk = cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

imgBrisk = cv2.drawKeypoints(gray, kps, None, color=(0,255,0), flags=0)
cv2.imshow("resultado", imgBrisk)
cv2.waitKey()
# 6. Finally, write the resulting image to the disk
cv2.imwrite('results//brisk_keypointlucca.jpg', imgBrisk)