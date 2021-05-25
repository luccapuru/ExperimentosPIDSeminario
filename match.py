import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('images//balliol_000025.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE) # trainImage
width = int(img2.shape[1] * 60 / 100)
height = int(img2.shape[0] * 60 / 100)
dim = (width, height)
img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
#img2 = cv.imread('images//balliol_000026.jpg',cv.IMREAD_GRAYSCALE)
#print(img1.shape())
# Initiate ORB detector
model = cv.BRISK_create(70, 4)
#model = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = model.detectAndCompute(img1,None)
kp2, des2 = model.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print("# keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
print("# keypoints2: {}, descriptors2: {}".format(len(kp2), des2.shape))
print(len(matches))
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('results\matchBRISK.jpg', img3)

cv.imshow('inicial', img3)
# cv.imshow('final', imgBrisk2)
cv.waitKey()