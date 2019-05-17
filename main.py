import os
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import time
import argparse
import tkinter as tk
import src.main_controller as controller

def plot_fit(x, y, func_fit):
    x_fit = np.array(range(0, int(max(x))))
    y_fit = exponential_function(x_fit, *func_fit)
    plt.plot(x, y, 'o', x_fit, y_fit, x_fit, y_fit2)
    plt.show()


window = tk.Tk()
window.title("ProcessMRI")

lbl = tk.Label(window, text="Load a MRI image: File/Open")
lbl.pack(side=tk.BOTTOM)

# open_button = tk.Button(window, text="Open...")
# open_button.pack(side=tk.TOP, expand=tk.YES)

# tpc_button = tk.Button(window, text="Correct phase")
# tpc_button.pack(side=tk.TOP, expand=tk.YES)

# density_button = tk.Button(window, text="Exponential fitting")
# density_button.pack(side=tk.TOP, expand=tk.YES)

window.geometry('350x200')



menu = tk.Menu(window)

open_menu = tk.Menu(window, tearoff=0)
open_menu.add_command(label="NifTi")
open_menu.add_command(label="Bruker directory")


file_menu = tk.Menu(menu, tearoff=0)
file_menu.add_cascade(label='Open', menu=open_menu)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)


process_menu = tk.Menu(menu, tearoff=0)
process_menu.add_command(label='Correct phase')
process_menu.add_command(label='Exponential fitting')


menu.add_cascade(label='File', menu=file_menu)
menu.add_cascade(label='Process', menu=process_menu)
window.config(menu=menu)

controller = controller.MainController(lbl, open_menu, process_menu)

window.mainloop()
