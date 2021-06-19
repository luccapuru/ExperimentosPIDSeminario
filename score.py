import numpy as np
import cv2
# import statistics as st

class ImageDivider():

    def Dividir(self, img, n): #numero de divisoes
        img_dividida = []
        k, j = 1, 1
        for i in range(n*4):
            img_dividida.append(img[(k-1)*(img.shape[0]//(n*2)):k*(img.shape[0]//(n*2)), 
            (j-1)*(img.shape[1]//(n*2)):j*(img.shape[1]//(n*2))])
            # print(i, (k-1)*(img.shape[0]//(n*2)))
            # print(i, k*(img.shape[0]//(n*2)))
            # print(i, (j-1)*(img.shape[1]//(n*2)))
            # print(i, j*(img.shape[1]//(n*2)))
            if k >= (n*2):
                k = 0
                j += 1
            k += 1
        return img_dividida
    
    def ShowImages(self, img_dividida):
        for i, imgd in enumerate(img_dividida):
            # print("-----------",imgd.shape)
            cv2.imshow("i" + str(i), imgd)
            # cv2.imwrite("results//corteImagem" + str(i)+ ".jpg", imgd)
    
    def MultipleBrisk(self, img_dividida):
        brisk = cv2.BRISK_create(70, 4)
        numberFeatures = []
        imgBrisk = []
        result = map(lambda img: brisk.detectAndCompute(img, None), img_dividida)
        kps = [x[:][0] for x in result]
        imgBrisk = map(lambda img, kp: cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0), img_dividida, kps)
        numberFeatures = [len(kp) for kp in kps]
        return imgBrisk, numberFeatures
    
    def FeatureDensity(self, height, width, numberFeatures):
        area = height*width
        density = [nF/area for nF in numberFeatures]
        # density = sum(density)/len(density)
        densityMean = np.mean(density)
        densityStd = np.std(density)
        return densityMean, densityStd
    
    def FeatureStatistic(self, numberFeatures):
        # featureMean = np.mean(numberFeatures)
        norm = np.linalg.norm(numberFeatures)
        numberFeatures = numberFeatures/norm
        featureStd = np.std(numberFeatures)
        return featureStd

    def Score(self, stdRef, stdValue, nFeatureRef, nFeature, contrastRef, contrastValue):
        #Distribuicao de Features
        distribution = stdRef/stdValue
        limit = lambda n, minn, maxn: max(min(maxn, n), minn)
        distribution = limit(distribution, 0, 1)

        #Numero de Features 
        qtd = nFeature/nFeatureRef
        qtd = limit(qtd, 0, 1)

        #contraste
        contrast = contrastValue/contrastRef
        contrast = limit(contrast, 0, 1)

        print("distribution", distribution)
        print("qtd", qtd)

    def RMSContrast(self, img):
        stdImg = np.std(img)
        return stdImg
    
    def MichContrast(self, img):
        print(np.max(img), np.min(img))
        contrast = (np.max(img)-np.min(img))/(np.max(img)+np.min(img))
        print(contrast)


def main():
    img = cv2.imread("images//dividir.png")
    n = input("Quantas divisões: ")
    divisor = ImageDivider()
    img_divida = divisor.Dividir(img, int(n))
    print(img.shape)
    # for i, imgd in enumerate(img_divida):
    #     print("-----------",imgd.shape)
    #     cv2.imshow("i" + str(i), imgd)
    #     cv2.imwrite("results//corteImagem" + str(i)+ ".jpg", imgd)

    cv2.waitKey()

if __name__ == "__main__":
    main()