import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import configparser
import os

class ExponentialFitView(tk.Frame):
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
        self.frame_header.grid(row=0, sticky="nw")

        self.label = tk.Label(self.frame_header, text="Exponential fit", font='Helvetica 14 bold')
        description = """Fit a n-exponential function on multiple echo data."""
        self.description = tk.Label(self.frame_header, text=description)
        self.label.grid(row=0, column=0, sticky="nw")
        self.description.grid(row=1, column=0, sticky="nw")



    def init_body(self):
        self.frame_body = tk.Frame(self)
        self.frame_body.grid(row=1, sticky="nw")

        self.label_method = tk.Label(self.frame_body, text="Fit method")
        self.label_threshold = tk.Label(self.frame_body, text="Threshold")
        self.label_destination = tk.Label(self.frame_body, text="Output directory")
        self.threshold = tk.Entry(self.frame_body, textvariable=tk.StringVar(self, "None")
        )

        self.compute_button = tk.Button(self.frame_body, text="Compute")
        self.choice_method = tk.ttk.Combobox(self.frame_body, values = [
            "Linear regression",
            "Mono-exponential",
            "Bi-exponential",
            "Tri-exponential"], state="readonly")
        self.choice_method.current(0)

        self.entry = tk.Entry(self.frame_body, textvariable=self.path)
        self.open_button = tk.Button(self.frame_body, command=self.open, text="Choose...")

        self.label_method.grid(row=2, column=0, sticky="nw")
        self.choice_method.grid(row=2, column=1, sticky="nw")
        self.label_threshold.grid(row=3, column=0, sticky="nw")
        self.threshold.grid(row=3, column=1, sticky="nw")
        self.label_destination.grid(row=4, column=0, sticky="nw")
        self.entry.grid(row=4, column=1, sticky="nw")
        self.open_button.grid(row=4, column=2, sticky="nw")
        self.compute_button.grid(row=5, column=2, sticky="se")

    def post_init(self):
        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=10)

    def open(self):
         directory = filedialog.askdirectory(parent=self, initialdir = self.config['default']['OutputDir'], title='Choose output directory')
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


    def show(self):
        self.grid()
        for widget in self.winfo_children():
            widget.grid()

    def hide(self):
        for widget in self.grid_slaves():
            widget.grid_remove()


if __name__ == "__main__":
    app = tk.Tk()
    ExponentialFitView(app, {})
    app.title("Exponential fit")
    app.mainloop()
