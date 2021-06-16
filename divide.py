import numpy as np
import cv2

class ImageDivider():

    def Dividir(self, img, n): #numero de divisoes
        img_dividida = []
        k, j = 1, 1
        for i in range(n*4):
            img_dividida.append(img[(k-1)*(img.shape[0]//(n*2)):k*(img.shape[0]//(n*2)), 
            (j-1)*(img.shape[1]//(n*2)):j*(img.shape[1]//(n*2))])
            print(i, (k-1)*(img.shape[0]//(n*2)))
            print(i, k*(img.shape[0]//(n*2)))
            print(i, (j-1)*(img.shape[1]//(n*2)))
            print(i, j*(img.shape[1]//(n*2)))
            if k >= (n*2):
                k = 0
                j += 1
            k += 1
        return img_dividida

def main():
    img = cv2.imread("images//dividir.png")
    n = input("Quantas divisões: ")
    divisor = ImageDivider()
    img_divida = divisor.Dividir(img, int(n))
    print(img.shape)
    for i, imgd in enumerate(img_divida):
        print("-----------",imgd.shape)
        cv2.imshow("i" + str(i), imgd)
        cv2.imwrite("results//corteImagem" + str(i)+ ".jpg", imgd)

    cv2.waitKey()

if __name__ == "__main__":
    main()