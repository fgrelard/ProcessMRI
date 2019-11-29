from src.qt.maincontroller import MainController
from src.qt.mainview import Ui_MainView
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import sys
import configparser
import os

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainView()
        self.ui.setupUi(self)
        self.show()


def init_configuration():
    """
    Initialisation of the configuration:
    storage preferences

    Parameters
    ----------
    None.

    Returns
    ----------
    configparser.ConfigParser:
        configuration preferences

    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    if 'default' not in config:
        config['default'] = {}
    if 'NifTiDir' not in config['default']:
        config['default']['NifTiDir'] = os.getcwd()
    return config

if __name__=='__main__':
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    main_window = AppWindow()
    config = init_configuration()
    main_controller = MainController(app, main_window.ui, config)
    main_window.show()
    app.exec_()
