import cv2
import numpy as np
import score 

divisor = score.ImageDivider()
img = cv2.imread("images//pedras-1.jpg", 0)
# img_dividida_ref = divisor.Dividir(img, 2)
# imgBrisk_ref, numberFeatures_ref = divisor.MultipleModel(img_dividida_ref, 2)
# print(numberFeatures_ref)
# print(np.sum(numberFeatures_ref))
# divisor.ShowImages(imgBrisk_ref)

# model = cv2.ORB_create(nfeatures = 500000, edgeThreshold = 7) #, patchSize = 7)
# model = cv2.ORB_create(nfeatures=100000)
model = cv2.BRISK_create(70, 4)
kps = model.detect(img,None)
# compute the descriptors with ORB
kps, descs = model.compute(img, kps)
print(kps[0].pt)
img2 = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
img_dividida_ref = divisor.Dividir(img2, 4)
divisor.ShowImages(img_dividida_ref)
divisor.SaveImages(img_dividida_ref)
print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
# print(divisor.FeatureDistribution(img, 1))
# print(np.sum(divisor.FeatureDistribution(img, 1)))
cv2.waitKey()

#[419. 639. 587. 388. 373. 528. 595. 354. 626. 604. 732. 463. 473. 427. 419. 288.]