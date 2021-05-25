import numpy as np
import cv2 
#from matplotlib import pyplot as plt
img = cv2.imread("images//lucca.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kps = orb.detect(img,None)
# compute the descriptors with ORB
kps, descs = orb.compute(img, kps)
print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

arr, uniq_cnt = np.unique(descs, axis=0, return_counts=True)
uniq_arr = arr[uniq_cnt==1]

#print(uniq_arr)
print("a", len(uniq_arr))

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(gray, kps, None, color=(0,255,0), flags=0)
#cv2.imwrite('orb_keypoints_high.jpg', img2)
cv2.imshow("resultado", img2)
cv2.waitKey()