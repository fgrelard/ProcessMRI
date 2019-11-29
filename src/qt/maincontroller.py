from src.qt.mainview import Ui_MainView
from PyQt5 import QtWidgets
import src.imageio as io
import os

class MainController:
    def __init__(self, app, mainview, config):
        self.mainview = mainview
        self.app = app
        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionNifti.triggered.connect(self.open_nifti)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.app.aboutToQuit.connect(self.exit_app)
        self.config = config
        self.images = {}

    def exit_app(self):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.app.quit()

    def choose_image(self, name):
        if name == "No image":
            return
        self.mainview.imageview.setImage(self.images[name])


    def open_nifti(self):
        """
        Opens nifti file and reads metadata
        """
        filedialog =  QtWidgets.QFileDialog()
        filename = filedialog.getOpenFileName(directory=self.config['default']['NifTiDir'],caption = "Select NifTi image")[0]
        print(filename)

        # getOpenFileName( filedialog.askopenfilename(initialdir = self.mainview.config['default']['NifTiDir'],title = "Select NifTi image",filetypes = (("nii files","*.nii*"),("all files","*.*")))
        try:
            self.config['default']['NifTiDir'] = os.path.dirname(filename)
            img = io.open_generic_image(filename)
        except Exception as e:
            print(e)
        else:
            self.filename = os.path.split(filename)[1]
            self.filename = self.filename.replace('.nii.gz', '')
            self.img_data = img.get_fdata()
            self.images[self.filename] = self.img_data
            self.mainview.imageview.setImage(self.img_data)
            self.mainview.combobox.addItem(self.filename)
        try:
            metadata = io.open_metadata(filename)
        except Exception as e:
            print("No metadata or echotimes")
            answer = QtWidgets.QInputDialog.getText(None, "No echotimes found", "Echotimes separated by a semi-colon", QtWidgets.QLineEdit.Normal, "")[0]
            echostring = answer.split(";")
            echostring = filter(None, echostring)
            echotime = [int(i) for i in echostring]
            self.echotime = echotime
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.echotime = echotime
