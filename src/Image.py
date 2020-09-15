import numpy as np

class Image(np.ndarray):
    def __new__(cls, input_array, contains_plot_info=False):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array, contains_plot_info=False):
        self.contains_plot_info = contains_plot_info
