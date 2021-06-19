import cv2
import score as s
import numpy as np

cc = s.ImageDivider()

# img = cv2.imread("images//leaves-contrast-high-rating.jpg", 0)
# img2 = cv2.imread("images//leaves-contrast-low-rating.jpg", 0)

# cc.RMSContrast(img)
# cc.RMSContrast(img2)
# cc.MichContrast(img)
# cc.MichContrast(img2)

img = np.zeros((200, 200))
img2 = np.ones((200, 200))*255
img3 = cv2.hconcat([img, img2])
print(cc.RMSContrast(img3))
cc.MichContrast(img3)
