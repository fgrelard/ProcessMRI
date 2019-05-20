import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import src.exponentialfitview as expview
import os

class MainView(tk.Tk):
    def __init__(self, config):
        tk.Tk.__init__(self)
        self.grid()
        self.init(config)

    def init(self, config):
        self.title("ProcessMRI")

        self.label = tk.Label(self, text="Load a MRI image: File/Open")
        self.geometry('500x500')

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

        self.menu.add_cascade(label='File', menu=self.file_menu)
        self.menu.add_cascade(label='Process', menu=self.process_menu)

        self.config(menu=self.menu)
        self.config = config
        self.progbar = ttk.Progressbar(self, orient="horizontal",
                                  length=200, mode="indeterminate")

        self.expframe = expview.ExponentialFitView(config)

        self.progbar.grid(row=1, sticky="se")
        self.hide_bar()

    def show_bar(self):
        self.progbar.grid()

    def hide_bar(self):
        self.progbar.grid_remove()

if __name__ == "__main__":
    app = MainView(None)
    app.mainloop()
