from src.qt.expfitcontroller import ExpFitController
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import src.imageio as io
import os
import numpy as np
import src.exponentialfit as expfit
from threading import Thread

class WorkerExpFit(QObject):

    signal_start = pyqtSignal()
    signal_end = pyqtSignal()

    def __init__(self, maincontroller, parent=None, threshold=None, lreg=True, n=1):
        super().__init__()
        self.maincontroller = maincontroller
        self.threshold = threshold
        self.lreg = lreg
        self.n = n

    @pyqtSlot()
    def work(self):
        print("workwork")
        self.maincontroller.app.processEvents()
        self.signal_start.emit()
        density, t2 = expfit.exponentialfit_image(self.maincontroller.echotime, self.maincontroller.img_data, self.threshold, self.lreg, self.n)
        self.signal_end.emit()
        self.maincontroller.add_image(density, "density")
        self.maincontroller.add_image(t2, "t2")
        self.maincontroller.choose_image("density")

class ThreadExpFit(Thread):
    def __init__(self, maincontroller, threshold, lreg, n):
        Thread.__init__(self)
        self.maincontroller = maincontroller
        self.threshold = threshold
        self.lreg = lreg
        self.n = n

    def run(self):
        self.maincontroller.mainview.show_run()
        density, t2 = expfit.exponentialfit_image(self.maincontroller.echotime, self.maincontroller.img_data, self.threshold, self.lreg, self.n)
        self.maincontroller.add_image(density, "density")
        self.maincontroller.add_image(t2, "t2")
        self.maincontroller.choose_image("density")
        self.maincontroller.mainview.hide_run()


class MainController:
    def __init__(self, app, mainview, config):
        self.mainview = mainview.ui
        self.mainview.parent = mainview
        self.app = app

        self.expfitcontroller = ExpFitController(mainview)
        self.expfitcontroller.signal.compute_signal.connect(self.exp_fit_estimation)

        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionBruker_directory.triggered.connect(self.open_bruker)
        self.mainview.actionNifti.triggered.connect(self.open_nifti)
        self.mainview.actionSave.triggered.connect(self.save_nifti)
        self.mainview.actionExponential_fitting.triggered.connect(self.expfitcontroller.show)
        self.mainview.actionDenoising_NL_means.triggered.connect(self.display_nl_means)
        self.mainview.actionDenoising_TPC.triggered.connect(self.display_tpc)

        # self.mainview.stopButton.clicked.connect(self.abort_workers)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.app.aboutToQuit.connect(self.exit_app)
        self.config = config
        self.images = {}
        self.mainview.hide_run()
        self.threads = []

    def open_bruker(self):
        """
        Opens Bruker directory
        """
        filedialog = QtWidgets.QFileDialog(None, "Select Bruker directory")
        filedialog.setOption(filedialog.DontUseNativeDialog)
        self.mainview.parent.move_dialog(filedialog)
        filedialog.setDirectory(self.config['default']['NifTiDir'])
        filedialog.setFileMode(filedialog.Directory)
        filedialog.setModal(False)
        if filedialog.exec():
            dirname = filedialog.selectedFiles()[0]
        try:
            list_filenames = io.open_generic_image(dirname)
        except Exception as e:
            print(e)
        else:
            if list_filenames:
                dirname = os.path.dirname(list_filenames[0])
                self.config['default']['NifTiDir'] = dirname
                self.open_nifti()

    def open_nifti(self):
        """
        Opens nifti file and reads metadata
        """
        filedialog = QtWidgets.QFileDialog(None, "Select Nifti")
        filedialog.setOption(filedialog.DontUseNativeDialog)
        self.mainview.parent.move_dialog(filedialog)
        filedialog.setDirectory(self.config['default']['NifTiDir'])
        if filedialog.exec():
            filename = filedialog.selectedFiles()[0]
        try:
            self.config['default']['NifTiDir'] = os.path.dirname(filename)
            img = io.open_generic_image(filename)
        except Exception as e:
            print(e)
        else:
            self.filename = os.path.split(filename)[1]
            self.filename = self.filename.replace('.nii', '')
            self.filename = self.filename.replace('.gz', '')
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


    def save_nifti(self):
        filedialog = QtWidgets.QFileDialog(None, "Save Nifti")
        filedialog.setOption(filedialog.DontUseNativeDialog)
        filedialog.setAcceptMode(filedialog.AcceptSave)
        self.mainview.parent.move_dialog(filedialog)
        filedialog.setDirectory(self.config['default']['NifTiDir'])
        if filedialog.exec():
            filename = filedialog.selectedFiles()[0]
            io.save_nifti_with_metadata(self.img_data, self.echotime, filename)

    def exit_app(self):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.app.quit()


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
                thread = ThreadExpFit(maincontroller=self, threshold=threshold, lreg=lreg, n=n)
                thread.start()
                # worker = WorkerExpFit(maincontroller=self, threshold=threshold, lreg=lreg, n=n)

                # thread = QThread()
                # worker.moveToThread(thread)
                # worker.signal_start.connect(self.mainview.show_run)
                # worker.signal_end.connect(self.mainview.hide_run)
                # thread.started.connect(worker.work)
                # thread.start()
                # self.threads.append((thread, worker))


    def display_nl_means(self):
        pass

    def display_tpc(self):
        pass

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

    def image_to_visualization(self, img):
        img2 = np.reshape(img, (img.shape[0], img.shape[1]) + (-1,), order='F')
        img2 = img2.transpose()
        return img2
