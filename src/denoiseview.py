import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import configparser
import os
import src.hoverview as hoverview

class DenoiseView(tk.Frame):

    def __init__(self, window, config):
        """
        Constructor of DenoiseView, inherits
        tkinter Frame

        Parameters
        ----------
        self: type
            description
        window: tk.Root
            main window
        config: configparser.Config
            configuration preferences
        """
        tk.Frame.__init__(self, window)
        self.grid()
        self.init(config)



    def init(self, config):
        """
        Various initialization functions
        """
        self.init_configuration(config)
        self.init_header()
        self.init_body()
        self.post_init()
        self.hide()
        self.show()


    def init_configuration(self, config):
        """
        Initializes the configuration preferences
        """
        self.config = config
        self.path = tk.StringVar(None)
        if 'default' not in self.config:
            self.config['default'] = {}
        if 'OutputDir' not in self.config['default']:
            self.config['default']['OutputDir'] = os.getcwd()
        self.path.set(self.config['default']['OutputDir'])

    def init_header(self):
        """
        Builds the header (title + description)
        """
        self.frame_header = tk.Frame(self)
        self.frame_header.grid(row=0, sticky="nw")

        self.label = tk.Label(self.frame_header, text="Rician Denoising", font='Helvetica 14 bold')
        description = """Denoise an image corrupted by Rician noise by a non-local means method."""
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="w")
        self.description.grid(row=1, column=0, sticky="nw")



    def init_body(self):
        """
        Builds the body (labels, entries, file chooser...)
        """
        self.frame_body = tk.Frame(self)
        self.frame_body.grid(row=1, sticky="nw")

        self.label_size = tk.Label(self.frame_body, text="Patch size")
        self.label_distance = tk.Label(self.frame_body, text="Patch distance")
        self.label_spread = tk.Label(self.frame_body, text="Noise spread")
        self.label_destination = tk.Label(self.frame_body, text="Output directory")
        self.size = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "5"))
        self.distance = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "6"))
        self.spread = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "1.5"))

        self.info_size = tk.Label(self.frame_body, text=" ? ", borderwidth=2, relief="groove")
        self.info_distance = tk.Label(self.frame_body, text=" ? ", borderwidth=2, relief="groove")
        self.info_spread = tk.Label(self.frame_body, text=" ? ", borderwidth=2, relief="groove")
        hoverview.HoverInfo(self.info_size, "Rectangular window of size n*n")
        hoverview.HoverInfo(self.info_distance, "Distance (in pixels) to search for similarity in intensities around a pixel")
        hoverview.HoverInfo(self.info_spread, "Multiplication factor of the estimated noise variance for noise correction")

        self.compute_button = tk.Button(self.frame_body, text="Compute")

        self.entry = tk.Entry(self.frame_body, textvariable=self.path)
        self.open_button = tk.Button(self.frame_body, command=self.open, text="Choose...")

        self.label_size.grid(row=2, column=0, sticky="sw")
        self.size.grid(row=2, column=1, sticky="sw")
        self.info_size.grid(row=2, column=2, sticky="sw")
        #h.grid(row=2, column=3, sticky="sw")
        self.label_distance.grid(row=3, column=0, sticky="sw")
        self.distance.grid(row=3, column=1, sticky="sw")
        self.info_distance.grid(row=3, column=2, sticky="sw")
        self.label_spread.grid(row=4, column=0, sticky="sw")
        self.spread.grid(row=4, column=1, sticky="sw")
        self.info_spread.grid(row=4, column=2, sticky="sw")
        self.label_destination.grid(row=5, column=0, sticky="sw")
        self.entry.grid(row=5, column=1, sticky="sw")
        self.open_button.grid(row=5, column=2, sticky="sw")
        self.compute_button.grid(row=6, column=2, sticky="se")

    def post_init(self):
        """
        Adds a padding of 10 for each element
        """
        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=10)

    def open(self):
        """
        Function called when the button for the choice of the directory is selected
        """
        directory = filedialog.askdirectory(parent=self, initialdir = self.config['default']['OutputDir'], title='Choose output directory')
        self.config['default']['OutputDir'] = directory
        self.path.set(directory)

    def show(self):
        """
        Shows this frame in the main window
        """
        self.grid()
        for widget in self.winfo_children():
            widget.grid()

    def hide(self):
        """
        Hides this frame in the main window
        """
        for widget in self.grid_slaves():
            widget.grid_remove()


if __name__ == "__main__":
    app = tk.Tk()
    DenoiseView(app, {})
    app.title("Denoising")
    app.mainloop()
