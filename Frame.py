import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import imageio
import score

class FrameImage():

    def __init__(self):
        self.scoreMaker = score.ImageDivider()

        #Start Interface
        self.root = tk.Tk()
        self.panelA = None
        self.panelB = None
        self.root.minsize(600, 400)
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

        #Referencias da imagem de referencia
        self.var.set(1)
        self.refImg, self.refTotalFeatures = self.OpenImg("C:/Users/lucca/Desktop/Mestrado/Disciplinas/Processamento de Imagens Digitais/Seminário/Experimentos/images/pedras-1.jpg")
        self.refContrast, self.refFeatureStd = self.CalcFeat(self.refImg) 
        # self.var.set(2)
        # self.refImg, self.refTotalFeaturesORB = self.OpenImg("C:/Users/lucca/Desktop/Mestrado/Disciplinas/Processamento de Imagens Digitais/Seminário/Experimentos/images/pedras-1.jpg")
        # self.refContrast, self.refFeatureStdBRISK = self.CalcFeat(self.refImg) 

        # self.refkps = self.LoadReference()
        self.root.mainloop()
    
    def OpenImg(self, path):
        if len(path) > 0:
            imgCV = imageio.imread(path)
            gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
            
            if self.var.get() == 1:
                model = cv2.BRISK_create(70, 4)
                print("BRISK")
            elif self.var.get() == 2:
                model = cv2.ORB_create(nfeatures=100000)
                print("ORB")
            (kps, descs) = model.detectAndCompute(gray, None)
            imgModel = cv2.drawKeypoints(imgCV, kps, None, color=(0,255,0), flags=0)

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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgDividida = self.scoreMaker.Dividir(img, 2) #divide a imagem em 8
        _, numberFeatures = self.scoreMaker.MultipleBrisk(imgDividida)
        featureStd = self.scoreMaker.FeatureStatistic(numberFeatures)
        contrast = self.scoreMaker.RMSContrast(gray)
        return contrast, featureStd

    def CalcScore(self):
        contrast, featureStd = self.CalcFeat(self.img)
        nota = self.scoreMaker.Score(self.refFeatureStd, featureStd, self.refTotalFeatures,
                self.totalFeatures, self.refContrast, contrast)
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("end", "Num de features: " + str(self.totalFeatures) + "\n" +
                                "Contraste RMS: " + str(contrast) + "\n" +
                                "Desvio Padrão das features: " + str(featureStd) + "\n" +
                                "Nota final: " + str(nota) + "/3.0")
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