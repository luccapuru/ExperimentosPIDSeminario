import tkinter as tk
from tkinter import filedialog as fd 
import cv2
import os
import imageio

try:
    import os
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
 
 
root = tk.Tk()
 
style = ttk.Style(root)
style.theme_use("clam")
 
 
def c_open_file_old():
    rep = filedialog.askopenfilenames(
    	parent=root,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
    		("PNG", "*.png"),
    		("JPEG", "*.jpg"),
    		("All files", "*")])
    # rep2 = rep[0].replace("/", "\\\\")
    rep2 = rep[0]
    print(os.path.isfile(rep2))
    print(rep2)
    img = imageio.imread(rep2)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    cv2.imshow("aaa", img)
    # try:
	#     os.startfile(rep[0])
    # except IndexError:
    #     print("No file selected")
 
ttk.Button(root, text="Open files", command=c_open_file_old).grid(row=1, column=0, padx=4, pady=4, sticky='ew')
 
root.mainloop()

# window = tk.Tk()
# label = tk.Label(
#     text="Hello, Tkinter",
#     fg="white",
#     bg="black",
#     width=10,
#     height=10
# )
# button = tk.Button(
#     text="Click me!",
#     width=25,
#     height=5,
#     bg="blue",
#     fg="yellow",
# )
# entry = tk.Entry(fg="yellow", bg="blue", width=50)
# label.pack()
# button.pack()
# entry.pack()
# entry.insert(0, "Python")
# print(entry.get())
# window.mainloop()