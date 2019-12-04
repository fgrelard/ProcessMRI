from src.qt.maincontroller import MainController
from src.qt.mainview import Ui_MainView
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import sys
import configparser
import os

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainView()
        self.ui.setupUi(self)
        self.show()

    def move_dialog(self, dialog):
        dialog_rect = dialog.mapToGlobal(dialog.rect().center())
        parent_rect = self.mapToGlobal(self.rect().center())
        dialog.move(parent_rect - dialog_rect)

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
    main_controller = MainController(app, main_window, config)
    main_window.show()
    app.exec_()
