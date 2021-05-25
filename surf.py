import cv2
import numpy as np

img = cv2.imread('house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)

kp, descs = surf.detectAndCompute(img,None)

kp, des = surf.detectAndCompute(img,None)

print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

arr, uniq_cnt = np.unique(descs, axis=0, return_counts=True)
uniq_arr = arr[uniq_cnt==1]

#print(uniq_arr)
print("a", len(uniq_arr))

img1 = cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('surf_keypoints.jpg', img2)
