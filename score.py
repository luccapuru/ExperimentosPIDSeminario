import numpy as np
import cv2
# import statistics as st

class ImageDivider():

    def Dividir(self, img, n): #numero de divisoes
        img_dividida = []
        k, j = 1, 1
        for i in range(n*4):
            img_dividida.append(img[(k-1)*(img.shape[0]//(n)):k*(img.shape[0]//(n)), 
            (j-1)*(img.shape[1]//(n)):j*(img.shape[1]//(n))])
            # print(i, (k-1)*(img.shape[0]//(n*2)))
            # print(i, k*(img.shape[0]//(n*2)))
            # print(i, (j-1)*(img.shape[1]//(n*2)))
            # print(i, j*(img.shape[1]//(n*2)))
            if k >= (n):
                k = 0
                j += 1
            k += 1
        return img_dividida
    
    def ShowImages(self, img_dividida):
        for i, imgd in enumerate(img_dividida):
            # print("-----------",imgd.shape)
            cv2.imshow("i" + str(i), imgd)
            # cv2.imwrite("results//corteImagem" + str(i)+ ".jpg", imgd)
    
    def SaveImages(self, img_dividida):
        for i, imgd in enumerate(img_dividida):
            # print("-----------",imgd.shape)
            cv2.imwrite("pedras//pedras" + str(i) + ".jpg", imgd)
            # cv2.imwrite("results//corteImagem" + str(i)+ ".jpg", imgd)
    
    def MultipleModel(self, img_dividida, var):
        if var == 1:
            # model = cv2.BRISK_create(20, 4)
            model = cv2.BRISK_create(70, 4)
        elif var == 2:
            model = cv2.ORB_create(nfeatures=100000)
            # model = cv2.ORB_create(nfeatures = 100000, edgeThreshold = 7) #, patchSize = 7)
        numberFeatures = []
        imgBrisk = []
        result = map(lambda img: model.detectAndCompute(img, None), img_dividida)
        kps = [x[:][0] for x in result]
        imgBrisk = map(lambda img, kp: cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0), img_dividida, kps)
        numberFeatures = [len(kp) for kp in kps]
        return imgBrisk, numberFeatures
    
    def FeatureDistribution(self, img, var):
        if var == 1:
            # model = cv2.BRISK_create(20, 4)
            model = cv2.BRISK_create(70, 4)
        elif var == 2:
            model = cv2.ORB_create(nfeatures=100000)
            # model = cv2.ORB_create(nfeatures = 100000, edgeThreshold = 7) #, patchSize = 7)
        kps, descs = model.detectAndCompute(img, None)
        numberFeatures = np.zeros(16)
        width, height = img.shape[1]//4, img.shape[0]//4
        # print(width, 2*width, 3*width, 4*width)
        # print(height, 2*height, 3*height, 4*height)
        # print(img.shape)
        # j, k = 1, 1
        for i in kps:
            if i.pt[1] < height and i.pt[0] < width:
                numberFeatures[0] += 1
            elif i.pt[1] < 2*height and i.pt[0] < width:
                numberFeatures[1] += 1
            elif i.pt[1] < 3*height and i.pt[0] < width:
                numberFeatures[2] += 1
            elif i.pt[1] < 4*height and i.pt[0] < width:
                numberFeatures[3] += 1

            elif i.pt[1] < height and i.pt[0] < 2*width:
                numberFeatures[4] += 1
            elif i.pt[1] < 2*height and i.pt[0] < 2*width:
                numberFeatures[5] += 1   
            elif i.pt[1] < 3*height and i.pt[0] < 2*width:
                numberFeatures[6] += 1
            elif i.pt[1] < 4*height and i.pt[0] < 2*width:
                numberFeatures[7] += 1  

            elif i.pt[1] < height and i.pt[0] < 3*width:
                numberFeatures[8] += 1
            elif i.pt[1] < 2*height and i.pt[0] < 3*width:
                numberFeatures[9] += 1   
            elif i.pt[1] < 3*height and i.pt[0] < 3*width:
                numberFeatures[10] += 1
            elif i.pt[1] < 4*height and i.pt[0] < 3*width:
                numberFeatures[11] += 1   

            elif i.pt[1] < 1*height and i.pt[0] < 4*width:
                numberFeatures[12] += 1
            elif i.pt[1] < 2*height and i.pt[0] < 4*width:
                numberFeatures[13] += 1   
            elif i.pt[1] < 3*height and i.pt[0] < 4*width:
                numberFeatures[14] += 1
            elif i.pt[1] < 4*height and i.pt[0] < 4*width:
                numberFeatures[15] += 1 

        print(numberFeatures)
        return numberFeatures                       

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
        print(numberFeatures)
        numberFeatures = numberFeatures/(norm+1)
        featureStd = np.std(numberFeatures)
        return featureStd

    def Score(self, stdRef, stdValue, nFeatureRef, nFeature, contrastRef, contrastValue):
        #Distribuicao de Features
        # distribution = 1 + 0.0591 - stdValue
        distribution = 1 + stdRef - stdValue
        limit = lambda n, minn, maxn: max(min(maxn, n), minn)
        distribution = limit(distribution, 0, 1)

        #Numero de Features 
        # qtd = nFeature/7915
        qtd = nFeature/nFeatureRef
        qtd = limit(qtd, 0, 1)

        #contraste
        # contrast = contrastValue/73.8385
        contrast = contrastValue/contrastRef
        contrast = limit(contrast, 0, 1)

        print("distribution", distribution)
        print("qtd", qtd)
        print("contrast", contrast)

        return contrast+qtd+distribution

    def RMSContrast(self, img):
        stdImg = np.std(img)
        return stdImg
    
    def MichContrast(self, img):
        print(np.max(img), np.min(img))
        contrast = (np.max(img)-np.min(img))/(np.max(img)+np.min(img))
        print(contrast)
    
    def ResizeImage(self, img, scalePercent = 80, maxW = 600 , maxH = 350):
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        dim = (int(img.shape[1] * scalePercent / 100), int(img.shape[0] * scalePercent / 100))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        while resized.shape[1] >= maxW or resized.shape[0] > maxH:
            dim = (int(resized.shape[1] * scalePercent / 100), int(resized.shape[0] * scalePercent / 100))
            resized = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)
        return resized


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