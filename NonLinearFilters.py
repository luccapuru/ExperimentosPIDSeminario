import cv2 
import numpy as np 

class NonLinearFilters():
    def Log(self, img):
        a = 255/np.log((1+np.ndarray.max(img)))
        imgLog = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        for i in range(imgLog.shape[0]):
            for j in range(imgLog.shape[1]):
                g = round(a*np.log(img[i,j] + 1))
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                imgLog[i,j] = g
        return imgLog

    def Exp(self, img):
        a = 255/np.log(1+np.ndarray.max(img)) #log inversa
        # a = 255/(np.e**np.ndarray.max(img) - 1)
        imgExp = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        for i in range(imgExp.shape[0]):
            for j in range(imgExp.shape[1]):
                g = round((np.e**img[i,j] - 1)**(1/a))
                #print(g)
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                imgExp[i,j] = g
        return imgExp

    def Quad(self, img):
        a = 255/ np.ndarray.max(img)**2
        imgQuad = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        for i in range(imgQuad.shape[0]):
            for j in range(imgQuad.shape[1]):
                g = a*img[i][j]**2
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                imgQuad[i,j] = g
        return imgQuad

    def Raiz(self, img):
        a = 255/ np.ndarray.max(img)**(1/2)
        imgRaiz = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        for i in range(imgRaiz.shape[0]):
            for j in range(imgRaiz.shape[1]):
                g = a*img[i][j]**(1/2)
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                imgRaiz[i,j] = g
        return imgRaiz

    def TwoStepTransform(self, img):
        aLog = 255/np.log((1+np.ndarray.max(img)))
        aExp = 255/np.log(1+np.ndarray.max(img))
        img2 = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                if img[i][j] < 128: #Exp
                    g = round((np.e**img[i,j] - 1)**(1/aExp))
                else: #Log
                    g = round(aLog*np.log(img[i,j] + 1))
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                img2[i,j] = g
        return img2
    
    def EqHist(self, img):
        img2 = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img2 = clahe.apply(img)
        return img2

def main():
    nlf = NonLinearFilters()
    #abrindo imagem em escala de cinza
    img = cv2.imread("images//all_souls_000005.jpg", cv2.IMREAD_GRAYSCALE)
    #mostrando a imagem orginal 
    cv2.imshow("Imagem Original", img)
    imgLog = nlf.Log(img)
    imgExp = nlf.Exp(img)
    imgQuad = nlf.Quad(img)
    imgRaiz = nlf.Raiz(img)
    imgEq = nlf.EqHist(img)
    imgTrans = nlf.TwoStepTransform(img)

    cv2.imshow("Imagem - Log", imgLog)
    cv2.imshow("Imagem - Exp", imgExp)
    cv2.imshow("Imagem - Quad", imgQuad)
    cv2.imshow("Imagem - Raiz", imgRaiz)
    cv2.imshow("Imagem - 2 passos", imgTrans)
    cv2.imshow("Imagem - Equalizacao de Histograma", imgEq)

    cv2.imwrite("results//Log.jpg", imgLog)
    cv2.imwrite("results//Exp.jpg", imgExp)
    cv2.imwrite("results//Quad.jpg", imgQuad)
    cv2.imwrite("results//Raiz.jpg", imgRaiz)
    cv2.imwrite("results//EqHist.jpg", imgEq)
    cv2.imwrite("results//TwoStep.jpg", imgTrans)

    cv2.waitKey()

if __name__ ==  "__main__":
    main()