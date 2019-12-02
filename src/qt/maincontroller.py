from src.qt.mainview import Ui_MainView
from PyQt5 import QtWidgets
import src.imageio as io
import os
import numpy as np

class MainController:
    def __init__(self, app, mainview, config):
        self.mainview = mainview
        self.app = app
        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionNifti.triggered.connect(self.open_nifti)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.app.aboutToQuit.connect(self.exit_app)
        self.mainview.imageview.scene.sigMouseMoved.connect(self.on_hover_image)
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

    def on_hover_image(self, evt):
        pos = evt  ## using signal proxy turns original arguments into a tuple
        imv = self.mainview.imageview
        mousePoint = imv.view.mapSceneToView(pos)
        x = int(mousePoint.x())
        y = int(mousePoint.y())
        image = imv.imageDisp
        if image is None:
            return
        if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[2]:
            t = imv.currentIndex
            imv.label.setText("<span>(%d, %d)</span><span style='font-size: 12pt; color: green;'>=%0.3f</span>" % (x, y, image[(t,y,x)]))


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
            self.mainview.combobox.addItem(self.filename)
            self.img_data = img.get_fdata()
            img_data_vis = img.get_fdata()
            img_data_vis = np.reshape(img_data_vis, (img_data_vis.shape[0], img_data_vis.shape[1]) + (-1,), order='F')
            img_data_vis = img_data_vis.transpose()
            self.images[self.filename] = img_data_vis
            self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(self.filename))
            self.choose_image(self.filename)
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
