import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import configparser
import os

class TemporalPhaseCorrectionView(tk.Frame):
    def __init__(self, window, config):
        tk.Frame.__init__(self, window)
        self.grid()
        self.init(config)

    def init(self, config):
        self.init_configuration(config)
        self.init_header()
        self.init_body()
        self.post_init()
        self.hide()
        self.show()


    def init_configuration(self, config):
        self.config = config
        self.path = tk.StringVar(None)
        if 'default' not in self.config:
            self.config['default'] = {}
        if 'OutputDir' not in self.config['default']:
            self.config['default']['OutputDir'] = os.getcwd()
        self.path.set(self.config['default']['OutputDir'])

    def init_header(self):
        self.frame_header = tk.Frame(self)
        self.frame_header.grid()

        self.label = tk.Label(self.frame_header, text="Temporal phase correction (TPC)", font='Helvetica 14 bold')
        description = """This tool allows to correct the phase from complex MRI images."""
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="w")
        self.description.grid(row=1, column=0, sticky="nw")



    def init_body(self):
        self.frame_body = tk.Frame(self)
        self.frame_body.grid()

        self.label_order = tk.Label(self.frame_body, text="Polynomial order")
        self.order = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "4")
        )
        self.label_destination = tk.Label(self.frame_body, text="Output directory")


        self.compute_button = tk.Button(self.frame_body, text="Compute")

        self.entry = tk.Entry(self.frame_body, textvariable=self.path)
        self.open_button = tk.Button(self.frame_body, command=self.open, text="Choose...")

        self.label_order.grid(row=2, column=0, sticky="sw")
        self.order.grid(row=2, column=1, sticky="sw")
        self.label_destination.grid(row=3, column=0, sticky="sw")
        self.entry.grid(row=3, column=1, sticky="sw")
        self.open_button.grid(row=3, column=2, sticky="sw")
        self.compute_button.grid(row=5, column=2, sticky="se")

    def post_init(self):
        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=10)

    def open(self):
         directory = filedialog.askdirectory(parent=self, initialdir =self.config['default']['OutputDir'], title='Choose output directory')
         self.config['default']['OutputDir'] = directory
         self.path.set(directory)


    def show(self):
        self.grid()
        for widget in self.winfo_children():
            widget.grid()

    def hide(self):
        for widget in self.grid_slaves():
            widget.grid_remove()


if __name__ == "__main__":
    app = tk.Tk()
    TemporalPhaseCorrectionView(app, {})
    app.title("Exponential fit")
    app.mainloop()
