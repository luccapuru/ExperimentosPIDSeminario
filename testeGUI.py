try:
    import os
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog

import cv2
import imageio
import tkcap
 
root = tk.Tk()
# canvas = Canvas(root, width = 300, height = 300)   
style = ttk.Style(root)
style.theme_use("clam")

def SaveScreen():
    cap = tkcap.CAP(root)
    cap.capture("screenshots//testeCap.jpg")
 
def OpenImg():
    rep = filedialog.askopenfilenames(
    	parent=root,
    	initialfile='tmp',
    	filetypes=[
    		("PNG", "*.png"),
    		("JPEG", "*.jpg"),
    		("All files", "*")])
    rep2 = rep[0]
    print(os.path.isfile(rep2))
    print(rep2)
    img = imageio.imread(rep2)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    cv2.imshow("aaa", img)
 
ttk.Button(root, text="Open files", command=OpenImg).grid(row=1, column=0, padx=4, pady=4, sticky='ew')
ttk.Button(root, text="Save Screen", command=SaveScreen).grid(row=2, column=0, padx=4, pady=4, sticky='ew')
 
root.mainloop()
