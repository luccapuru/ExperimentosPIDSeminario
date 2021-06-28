import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import imageio
import score
import tkcap
from PIL import ImageGrab
from functools import partial
import numpy as np
ImageGrab.grab = partial(ImageGrab.grab, all_screens=True)

class FrameImage():

    def __init__(self):
        self.scoreMaker = score.ImageDivider()

        #Start Interface
        self.root = tk.Tk()
        self.root.title("Marker Score Calculator")
        self.root.iconbitmap("images//1321.ico")
        self.panelA = None
        self.panelB = None
        self.root.minsize(600, 400)
        self.btn3 = tk.Button(self.root, text="Save Screen", command=self.SaveScreen)
        self.btn3.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        self.btn2 = tk.Button(self.root, text="Calculate Score", command=self.CalcScore)
        self.btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        self.btn2.configure(state="disabled")
        self.btn = tk.Button(self.root, text="Select an image", command=self.select_image)
        self.btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        self.var = tk.IntVar()
        self.R1 = tk.Radiobutton(self.root, text="BRISK", variable=self.var, value=1)
        self.R1.pack()
        self.R2 = tk.Radiobutton(self.root, text="ORB", variable=self.var, value=2)
        self.R2.pack()
        self.text = tk.Text(self.root, state='disabled', width=44, height=5)
        self.text.pack()

        self.var.set(2)
        self.refImg, self.refTotalFeaturesORB = self.OpenImg("C:/Users/lucca/Desktop/Mestrado/Disciplinas/Processamento de Imagens Digitais/Seminário/Experimentos/images/pedras-1.jpg")
        self.refContrast, self.refFeatureStdORB = self.CalcFeat(self.refImg) 
        #Referencias da imagem de referencia
        self.var.set(1)
        self.refImg, self.refTotalFeaturesBRISK = self.OpenImg("C:/Users/lucca/Desktop/Mestrado/Disciplinas/Processamento de Imagens Digitais/Seminário/Experimentos/images/pedras-1.jpg")
        self.refContrast, self.refFeatureStdBRISK = self.CalcFeat(self.refImg) 
        
        # self.refkps = self.LoadReference()
        self.root.mainloop()
    
    def SaveScreen(self):
        cap = tkcap.CAP(self.root)
        cap.capture("screenshots//Cap.jpg")
    
    def OpenImg(self, path):
        if len(path) > 0:
            imgCV = imageio.imread(path)
            if len(imgCV.shape) > 2:
                gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
            else:
                gray = np.copy(imgCV)
            if self.var.get() == 1:
                # model = cv2.BRISK_create(20, 4)
                model = cv2.BRISK_create(70, 4)
                print("BRISK")
                (kps, descs) = model.detectAndCompute(gray, None)
                imgModel = cv2.drawKeypoints(imgCV, kps, None, color=(0,255,0), flags=0)
            elif self.var.get() == 2:
                model = cv2.ORB_create(nfeatures=100000)
                print("ORB")
                (kps, descs) = model.detectAndCompute(gray, None)
                imgModel = cv2.drawKeypoints(imgCV, kps, None, color=(0,0,255), flags=0)
            
            imgTK = self.scoreMaker.ResizeImage(imgCV)
            imgModel = self.scoreMaker.ResizeImage(imgModel)

            imgTK = Image.fromarray(imgTK); imgTK = ImageTk.PhotoImage(imgTK); 
            imgModel = Image.fromarray(imgModel); imgModel = ImageTk.PhotoImage(imgModel); 

        if self.panelA is None or self.panelB is None:
            self.panelA = tk.Label(image=imgTK)
            self.panelA.image = imgTK
            self.panelA.pack(side="left", padx=10, pady=10)

            self.panelB = tk.Label(image=imgModel)
            self.panelB.image = imgModel
            self.panelB.pack(side="right", padx=10, pady=10)
        else:
            self.panelA.configure(image=imgTK)
            self.panelB.configure(image=imgModel)
            self.panelA.image = imgTK
            self.panelB.image = imgModel
        # print(type(img))
        return imgCV, len(kps)

    def CalcFeat(self, img):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.copy(img)
        # imgDividida = self.scoreMaker.Dividir(img, 2) #divide a imagem em 8
        # _, numberFeatures,  = self.scoreMaker.MultipleModel(imgDividida, self.var.get())
        numberFeatures = self.scoreMaker.FeatureDistribution(img, self.var.get())
        featureStd = self.scoreMaker.FeatureStatistic(numberFeatures)
        contrast = self.scoreMaker.RMSContrast(gray)
        return contrast, featureStd

    def CalcScore(self):
        contrast, featureStd = self.CalcFeat(self.img)
        if self.var.get() == 1:
            nota = self.scoreMaker.Score(self.refFeatureStdBRISK, featureStd, self.refTotalFeaturesBRISK,
                self.totalFeatures, self.refContrast, contrast)
        elif self.var.get() == 2:
            nota = self.scoreMaker.Score(self.refFeatureStdORB, featureStd, self.refTotalFeaturesORB,
                self.totalFeatures, self.refContrast, contrast)
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("end", "Number of Features: " + str(self.totalFeatures) + "\n" +
                                "RMS Contrast: " + "{:.4f}".format(contrast) + "\n" +
                                "Feature Distribution: " + "{:.4f}".format(featureStd) + "\n" +
                                "Final Score: " + "{:.2f}".format(nota) + "/3.00")
        self.text.configure(state="disabled")
        self.btn2.configure(state="disabled")

    def select_image(self):
        path = filedialog.askopenfilenames()
        self.img, self.totalFeatures = self.OpenImg(path[0])
        self.btn2.configure(state="normal")
    

def main():
    janela = FrameImage()

if __name__ == "__main__":
    main()