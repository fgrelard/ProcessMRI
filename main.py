import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import tkinter as tk
import src.main_controller as controller
import src.mainview as view
import src.exponentialfitview as expview
from tkinter import ttk
import configparser

def init_configuration():
    """
    Initialisation of the configuration:
    storage preferences

    Parameters
    ----------
    None.

    Returns
    ----------
    config

    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    if 'default' not in config:
        config['default'] = {}
    if 'NifTiDir' not in config['default']:
        config['default']['NifTiDir'] = os.getcwd()
    return config

config = init_configuration()
mainview = view.MainView(config)
controller = controller.MainController(mainview)
mainview.mainloop()
