import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import configparser
import os
import src.hoverview as hoverview

class TemporalPhaseCorrectionView(tk.Frame):
    def __init__(self, window):
        """
        Constructor of TemporalPhaseCorrectionView
        inherits tkinter Frame

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
        self.init()

    def init(self):
        """
        Various initialization functions
        """
        self.init_header()
        self.init_body()
        self.post_init()
        self.hide()
        self.show()



    def init_header(self):
        """
        Builds the header (title + description)
        """
        self.frame_header = tk.Frame(self)
        self.frame_header.grid(row=0, sticky="nw")

        self.label = tk.Label(self.frame_header, text="Temporal phase correction (TPC)", font='Helvetica 14 bold')
        self.info_label = tk.Label(self.frame_header, text=" ? ", borderwidth=2, relief="groove")
        description = """Correct the phase from complex MRI images,\n in order to obtain a strictly gaussian noise distribution."""
        hoverview.HoverInfo(self.info_label, """The denoised signal is found in the real part of the temporally corrected image,\nand the noise is in the imaginary image.""")
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="w")
        self.info_label.grid(row=0, column=1, sticky="w")
        self.description.grid(row=1, column=0, sticky="nw")

    def init_body(self):
        """
        Builds the body (labels, entries, file chooser...)
        """
        self.frame_body = tk.Frame(self)
        self.frame_body.grid(row=1, sticky="nw")

        self.label_order = tk.Label(self.frame_body, text="Polynomial order")
        self.order = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "4")
        )
        self.info_order = tk.Label(self.frame_body, text=" ? ", borderwidth=2, relief="groove")
        hoverview.HoverInfo(self.info_order, "Order of the fitted polynomial to correct phase")

        self.label_noise = tk.Label(self.frame_body, text="Noise threshold")
        self.noise = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "0")
        )
        self.info_noise = tk.Label(self.frame_body, text=" ? ", borderwidth=2, relief="groove")
        hoverview.HoverInfo(self.info_noise, "Noise threshold on magnitude to discard air pixels")


        self.compute_button = tk.Button(self.frame_body, text="Compute")


        self.label_order.grid(row=2, column=0, sticky="sw")
        self.order.grid(row=2, column=1, sticky="sw")
        self.info_order.grid(row=2, column=2, sticky="sw")
        self.label_noise.grid(row=3, column=0, sticky="sw")
        self.noise.grid(row=3, column=1, sticky="sw")
        self.info_noise.grid(row=3, column=2, sticky="sw")
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
        directory = filedialog.askdirectory(parent=self, initialdir =self.config['default']['OutputDir'], title='Choose output directory')
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
    TemporalPhaseCorrectionView(app, {})
    app.title("Exponential fit")
    app.mainloop()
