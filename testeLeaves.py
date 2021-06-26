import cv2
import numpy as np
import score

img = cv2.imread("images//leaves-contrast-low-rating.jpg")
divisor = score.ImageDivider()
img_dividida = divisor.Dividir(img, 2)
imgBrisk, numberFeatures = divisor.MultipleModel(img_dividida, 2)
divisor.ShowImages(imgBrisk)
cv2.waitKey()