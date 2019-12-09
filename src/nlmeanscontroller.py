from src.nlmeansview import Ui_NLMeans_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.exponentialfit as expfit


class WorkerNLMeans(QtCore.QObject):
    """
    Worker class for the NL means denoising
    Instances of this class can be moved to a thread

    Attributes
    ----------
    img_data: np.ndarray
        the image
    patch_size: int
        size of the patch n*n
    patch_distance: int
        distance in pixels to search for patch
    noise_spread: float
        multiplication factor for estimated noise variance
    is_abort: bool
        whether the computation was aborted and should be stopped
    """

    #PyQt5 signals
    #Signal emitted at the start of the computation
    signal_start = QtCore.pyqtSignal()

    #Signal emitted at the end of the computation
    signal_end = QtCore.pyqtSignal(np.ndarray, int)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, patch_size, patch_distance, noise_spread):
        super().__init__()
        self.img_data = img_data
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.noise_spread = noise_spread
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of NL means denoising

        Analogous to expfit.denoise_image
        """
        self.signal_start.emit()
        denoised = np.zeros_like(self.img_data)
        dim = len(self.img_data.shape)
        if dim > 3:
            length = self.img_data.shape[-1]
            for i in range(self.img_data.shape[-1]):
                if self.is_abort:
                    break
                QApplication.processEvents()
                image3D = expfit.denoise_2_3D(self.img_data[...,i], self.patch_size, self.patch_distance, self.noise_spread)
                denoised[..., i] = image3D
                progress = float(i / length * 100)
                self.signal_progress.emit(progress)
        else:
            denoised = expfit.denoise_2_3D(self.img_data, self.patch_size, self.patch_distance, self.noise_spread)
        if not self.is_abort:
            self.signal_end.emit(denoised, WorkerNLMeans.number)
            WorkerNLMeans.number += 1


    def abort(self):
        self.is_abort = True

class NLMeansController:
    """
    Controller handling the NLMeansView dialog

    Attributes
     ----------
     view: Ui_NLMeans_View
        the view
     trigger: Signal
        signal raised when clicking on the "OK" button
     """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_NLMeans_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)
        self.view.pushButton.setFixedWidth(20)
        self.view.pushButton_2.setFixedWidth(20)
        self.view.pushButton_3.setFixedWidth(20)

        #Tooltips
        t1 = self.view.pushButton.toolTip()
        t2 = self.view.pushButton_2.toolTip()
        t3 = self.view.pushButton_3.toolTip()
        self.view.pushButton.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t1)
        self.view.pushButton_2.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t2)
        self.view.pushButton_3.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t3)

        #Reset tooltips to avoid overlap of events
        self.view.pushButton.setToolTip("")
        self.view.pushButton_2.setToolTip("")
        self.view.pushButton_3.setToolTip("")

        #Events
        self.trigger = Signal()
        self.view.buttonBox.accepted.connect(self.update_parameters)


    def update_parameters(self):
        """
        Gets the values in the GUI and updates the attributes
        """
        self.patch_size = self.view.lineEdit.text()
        self.patch_distance = self.view.lineEdit_2.text()
        self.noise_spread = self.view.lineEdit_3.text()
        self.trigger.signal.emit()


    def show(self):
        self.dialog.show()
