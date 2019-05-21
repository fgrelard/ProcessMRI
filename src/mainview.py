import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import src.exponentialfitview as expview
import src.temporalphasecorrectionview as tpcview
import src.denoiseview as denoiseview
import os

class MainView(tk.Tk):
    def __init__(self, config):
        tk.Tk.__init__(self)
        self.grid()
        self.init(config)

    def init(self, config):
        self.title("ProcessMRI")

        self.menu = tk.Menu(self)

        self.open_menu = tk.Menu(self, tearoff=0)
        self.open_menu.add_command(label="NifTi")
        self.open_menu.add_command(label="Bruker directory")


        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.file_menu.add_cascade(label='Open', menu=self.open_menu)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit")


        self.process_menu = tk.Menu(self.menu, tearoff=0)
        self.process_menu.add_command(label='Exponential fitting')
        self.process_menu.add_command(label='Correct phase')
        self.process_menu.add_command(label='Denoise')

        self.menu.add_cascade(label='File', menu=self.file_menu)
        self.menu.add_cascade(label='Process', menu=self.process_menu)

        self.config(menu=self.menu)
        self.config = config
        self.progbar = ttk.Progressbar(self, orient="horizontal",
                                  length=100, mode="indeterminate")

        self.frames = {}
        self.expframe = expview.ExponentialFitView(self, config)
        self.tpcframe = tpcview.TemporalPhaseCorrectionView(self, config)
        self.denoiseframe = denoiseview.DenoiseView(self, config)
        self.frame()

        self.frames[MainView.__name__] = self.frame_header
        self.frames[expview.ExponentialFitView.__name__] = self.expframe
        self.frames[tpcview.TemporalPhaseCorrectionView.__name__] = self.tpcframe
        self.frames[denoiseview.DenoiseView.__name__] = self.denoiseframe

        self.frame_header.grid(row=0, column=0, sticky="nsew")
        self.expframe.grid(row=0, column=0, sticky="nsew")
        self.tpcframe.grid(row=0, column=0, sticky="nsew")
        self.denoiseframe.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainView")

        self.back_button = tk.Button(self, text="Back")
        self.back_button.grid(row=1, column=0, sticky="sw")
        self.progbar.grid(row=1, column=0,sticky="se")

        self.post_init()

        self.hide_bar()


    def post_init(self):
        for child in self.winfo_children():
            try:
                child.grid_configure(padx=10, pady=10)
            except:
                pass


    def frame(self):
        self.frame_header = tk.Frame(self)
        self.frame_header.grid()

        self.label = tk.Label(self.frame_header, text="Process MRI", font='Helvetica 14 bold')
        description = """Open an image (File/Open...)"""
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="w")
        self.description.grid(row=1, column=0, sticky="nw")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def show_bar(self):
        self.progbar.grid()

    def hide_bar(self):
        self.progbar.grid_remove()

if __name__ == "__main__":
    app = MainView(None)
    app.mainloop()