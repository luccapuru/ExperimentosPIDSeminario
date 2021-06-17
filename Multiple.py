import cv2
import numpy as np
import divide 


def main():
    # img = cv2.imread("images//all_souls_000005.jpg", 0)
    img = cv2.imread("images//pedras-1.jpg", 0)
    print(img.shape)
    # cv2.imshow("img", img)
    divisor = divide.ImageDivider()
    img_dividida = divisor.Dividir(img, 2)
    # divisor.ShowImages(img_dividida)
    # imgBrisk, numberFeatures = divisor.MultipleBrisk(img_dividida)
    imgBrisk, numberFeatures = divisor.MultipleBrisk(img_dividida)
    print(numberFeatures)
    divisor.ShowImages(imgBrisk)
    density = divisor.FeatureDensity(img_dividida[0].shape[0], img_dividida[0].shape[1], numberFeatures)
    print(density)


if __name__ == "__main__":
    main()
    cv2.waitKey()