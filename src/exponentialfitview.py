import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import configparser
import os

class ExponentialFitView(tk.Tk):
    def __init__(self, config):
        tk.Tk.__init__(self)
        self.init(config)

    def init(self, config):
        self.init_configuration(config)
        self.init_header()
        self.init_body()

    def init_header(self):
        self.frame_header = tk.Frame(self)
        self.frame_header.grid()

        self.label = tk.Label(self.frame_header, text="Exponential fit", font='Helvetica 14 bold')
        description ="""This tool allows to fit a n-exponential function on multiple echo data."""
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="w")
        self.description.grid(row=1, column=0, sticky="nw")

    def init_configuration(self, config):
        self.config = config
        if not self.config:
            return
        if 'default' not in self.config or 'OutputDir' not in self.config['default']:
            self.config['default'] = {}
            self.config['default']['OutputDir'] = os.getcwd()


    def init_body(self):
        self.frame_body = tk.Frame(self)
        self.frame_body.grid()

        self.label_method = tk.Label(self.frame_body, text="Fit method")
        self.label_threshold = tk.Label(self.frame_body, text="Threshold")
        self.label_destination = tk.Label(self.frame_body, text="Output directory")
        self.threshold = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "0") )

        self.compute_button = tk.Button(self.frame_body, text="Compute", command=self.quit)
        self.choice_method = tk.ttk.Combobox(self.frame_body, values = [
            "Linear regression",
            "Mono-exponential",
            "Bi-exponential",
            "Tri-exponential"], state="readonly")

        self.path = tk.StringVar(None)
        self.entry = tk.Entry(self.frame_body, textvariable=self.path)
        self.open_button = tk.Button(self.frame_body, command=self.open, text="Choose...")

        self.label_method.grid(row=2, column=0, sticky="sw")
        self.choice_method.grid(row=2, column=1, sticky="sw")
        self.label_threshold.grid(row=3, column=0, sticky="sw")
        self.threshold.grid(row=3, column=1, sticky="sw")
        self.label_destination.grid(row=4, column=0, sticky="sw")
        self.entry.grid(row=4, column=1, sticky="sw")
        self.open_button.grid(row=4, column=2, sticky="sw")
        self.compute_button.grid(row=5, column=2, sticky="se")


    def open(self):
         directory = filedialog.askdirectory(parent=self,title='Choose output directory')
         self.config['default']['OutputDir'] = directory
         self.path.set(directory)

    def validate_int(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if text in '0123456789.-+':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

if __name__ == "__main__":
    app = ExponentialFitView(None)
    app.title("Exponential fit")
    app.mainloop()
