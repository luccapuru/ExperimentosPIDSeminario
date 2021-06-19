import cv2
import score

#-----Reading the image-----------------------------------------------------
img = cv2.imread('images//all_souls_000005.jpg')
# cv2.imshow("img",img) 

# #-----Converting image to LAB Color model----------------------------------- 
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow("lab",lab)

# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
# # cv2.imshow('l_channel', l)
# # cv2.imshow('a_channel', a)
# # cv2.imshow('b_channel', b)

# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# # cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,a,b))
# # cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# # cv2.imshow('final', final)
# # cv2.waitKey()
d = score.ImageDivider()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
final = clahe.apply(gray)
cv2.imwrite("images//contrastgray.jpg", gray)
cv2.imwrite("images//contrasteq.jpg", final)
cv2.imwrite("images//contrast.jpg", cv2.hconcat([gray, final]))

d.RMSContrast(gray)
d.RMSContrast(final)
d.MichContrast(gray)
d.MichContrast(final)

#gray2 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
model = cv2.BRISK_create(20, 4)
#model = cv2.ORB_create(nfeatures=100000)
(kps, descs) = model.detectAndCompute(gray, None)
(kps2, descs2) = model.detectAndCompute(final, None)
print("# keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
print("# keypoints2: {}, descriptors2: {}".format(len(kps2), descs2.shape))
# imgBrisk = cv2.drawKeypoints(gray, kps, None, color=(255,0,0), flags=0)
# imgBrisk2 = cv2.drawKeypoints(final, kps2, None, color=(255,0,0), flags=0)

imgBrisk = cv2.drawKeypoints(gray, kps, None, color=(0,255,0), flags=0)
imgBrisk2 = cv2.drawKeypoints(final, kps2, None, color=(0,255,0), flags=0)

imgBrisk3 = cv2.hconcat([imgBrisk, imgBrisk2])
# cv2.imwrite('results//contrastOrb1.jpg', imgBrisk)
# cv2.imwrite('results//contrastOrb2.jpg', imgBrisk2)
# cv2.imwrite("results//contrastORB.jpg", imgBrisk3)

cv2.imwrite('results//contrastBRISK1.jpg', imgBrisk)
cv2.imwrite('results//contrastBRISK2.jpg', imgBrisk2)
cv2.imwrite("results//contrastBRISK.jpg", imgBrisk3)

cv2.imshow('inicial', imgBrisk)
cv2.imshow('final', imgBrisk2)
cv2.imshow("teste", imgBrisk3)
cv2.waitKey()
#_____END_____#