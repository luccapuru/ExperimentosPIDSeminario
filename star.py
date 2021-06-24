from tkinter import *

canvas_width = 200
canvas_height =200
python_green = "#476042"

master = Tk()

w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack()

points = [10,40,40,40,50,10,60,40,90,40,65,60,75,90,50,70,25,90,35,60]

w.create_polygon(points, outline=python_green, 
            fill='yellow', width=1)

mainloop()