from src.qt.expfitcontroller import ExpFitController
from PyQt5 import QtWidgets
import src.imageio as io
import os
import numpy as np
import src.exponentialfit as expfit


class MainController:
    def __init__(self, app, mainview, config):
        self.mainview = mainview.ui
        self.app = app

        self.expfitcontroller = ExpFitController(mainview)
        self.expfitcontroller.signal.compute_signal.connect(self.exp_fit_estimation)

        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionNifti.triggered.connect(self.open_nifti)
        self.mainview.actionExponential_fitting.triggered.connect(self.expfitcontroller.show)
        self.mainview.actionDenoising_NL_means.triggered.connect(self.display_nl_means)
        self.mainview.actionDenoising_TPC.triggered.connect(self.display_tpc)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.app.aboutToQuit.connect(self.exit_app)
        self.mainview.imageview.scene.sigMouseMoved.connect(self.on_hover_image)
        self.config = config
        self.images = {}
        self.mainview.hide_run()

    def exp_fit_estimation(self):
        """
        Density and T2 estimation from
        exponential fitting
        see expfit.exponentialfit_image
        """
        fit_method = self.expfitcontroller.fit_method
        threshold = self.expfitcontroller.threshold
        outname = self.config['default']['NifTiDir']
        print(self.img_data.shape)
        if self.img_data is not None:
            try:
                threshold = int(threshold)
            except:
                print("Automatic threshold with gaussian mixture")
                threshold = None
            finally:
                lreg = True
                n=1
                if fit_method != "Linear regression":
                    lreg = False
                    if fit_method == "Mono-exponential":
                        n=1
                    elif fit_method == "Bi-exponential":
                        n=2
                    else:
                        n=3
                self.mainview.show_run()
                density, t2 = expfit.exponentialfit_image(self.echotime, self.img_data, threshold, lreg, n)
                self.add_image(density, "density")
                self.add_image(t2, "t2")
                self.choose_image("density")
                self.mainview.hide_run()


    def display_nl_means(self):
        pass

    def display_tpc(self):
        pass

    def exit_app(self):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.app.quit()

    def add_image(self, image, name):
        self.mainview.combobox.addItem(name)
        self.images[name] = image

    def choose_image(self, name):
        if name == "No image":
            return
        if name not in self.images:
            return
        self.img_data = self.images[name]
        self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(name))
        vis = self.image_to_visualization(self.img_data)
        self.mainview.imageview.setImage(vis)

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

    def image_to_visualization(self, img):
        img2 = np.reshape(img, (img.shape[0], img.shape[1]) + (-1,), order='F')
        img2 = img2.transpose()
        return img2

    def open_nifti(self):
        """
        Opens nifti file and reads metadata
        """
        filename, _ =  QtWidgets.QFileDialog.getOpenFileName(None, directory=self.config['default']['NifTiDir'],caption = "Select NifTi image")

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
            self.add_image(img.get_fdata(), self.filename)
            self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(self.filename))
            self.choose_image(self.filename)
        try:
            metadata = io.open_metadata(filename)
        except Exception as e:
            print("No metadata or echotimes")
            answer, _ = QtWidgets.QInputDialog.getText(None, "No echotimes found", "Echotimes separated by a semi-colon", QtWidgets.QLineEdit.Normal, "")
            echostring = answer.split(";")
            echostring = filter(None, echostring)
            echotime = [int(i) for i in echostring]
            self.echotime = echotime
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.echotime = echotime
