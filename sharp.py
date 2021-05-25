import cv2
import numpy as np
img = cv2.imread('images//all_souls_000099.jpg')
sharpen_kernel = np.array([[-1,-1,-1], 
                           [-1,9,-1], 
                           [-1,-1,-1]])
final = cv2.filter2D(img, -1, sharpen_kernel)

# cv2.imshow('sharpen', sharpen)
# cv2.waitKey()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
cv2.imwrite("images//sharpengray.jpg", gray)
cv2.imwrite("images//sharpened.jpg", final)
cv2.imwrite("images//sharp.jpg", cv2.hconcat([gray, final]))

model = cv2.BRISK_create(70, 4)
#model = cv2.ORB_create(nfeatures=100000)
(kps, descs) = model.detectAndCompute(gray, None)
(kps2, descs2) = model.detectAndCompute(final, None)
print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
print("# keypoints2: {}, descriptors2: {}".format(len(kps2), descs2.shape))

# imgBrisk = cv2.drawKeypoints(gray, kps, None, color=(255,0,0), flags=0)
# imgBrisk2 = cv2.drawKeypoints(final, kps2, None, color=(255,0,0), flags=0)

imgBrisk2 = cv2.drawKeypoints(final, kps2, None, color=(0,255,0), flags=0)
imgBrisk = cv2.drawKeypoints(gray, kps, None, color=(0,255,0), flags=0)

imgBrisk3 = cv2.hconcat([imgBrisk, imgBrisk2])
cv2.imwrite('results//sharpBrisk1.jpg', imgBrisk)
cv2.imwrite('results//sharpBrisk2.jpg', imgBrisk2)
cv2.imwrite("results//sharpBRISK.jpg", imgBrisk3)

# cv2.imwrite('results//sharpORB1.jpg', imgBrisk)
# cv2.imwrite('results//sharpORB2.jpg', imgBrisk2)
# cv2.imwrite("results//sharpORB.jpg", imgBrisk3)

cv2.imshow('inicial', imgBrisk)
cv2.imshow('final', imgBrisk2)
cv2.waitKey()