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

# parser = argparse.ArgumentParser(description='Converts a 4D MRI image with multiple echoes for each pixel to a 3D density image obtained by considering the magnitude extrapolated at tau=0 by fitting a n-exponential on the echo magnitudes')
# parser.add_argument("-i", "--input", help="Input 4D MRI image (x,y,z,echoes), extension: NIFTI)")
# parser.add_argument("-o", "--output", help="Output 3D density image (x,y,z), extension: NIFTI")
# parser.add_argument("-t", "--threshold", help="Threshold for noise")
# args = parser.parse_args()

window = tk.Tk()
window.title("ProcessMRI")

lbl = tk.Label(window, text="Choose a MRI image file before going further")
lbl.pack(side=tk.BOTTOM)

open_button = tk.Button(window, text="Open...")
open_button.pack(side=tk.TOP, expand=tk.YES)

tpc_button = tk.Button(window, text="Correct phase")
tpc_button.pack(side=tk.TOP, expand=tk.YES)

density_button = tk.Button(window, text="Exponential fitting")
density_button.pack(side=tk.TOP, expand=tk.YES)

window.geometry('350x200')

menu = tk.Menu(window)
file_item = tk.Menu(menu)
file_item.add_command(label='Open...')
process_item = tk.Menu(menu)
process_item.add_command(label='Correct phase')
process_item.add_command(label='Exponential fitting')
menu.add_cascade(label='File', menu=file_item)
menu.add_cascade(label='Process', menu=process_item)
window.config(menu=menu)

controller = controller.MainController(lbl, open_button, tpc_button, density_button, menu)

window.mainloop()

# input = args.input
# output = args.output
# threshold = args.threshold
