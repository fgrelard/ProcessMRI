from src.manualcomponentview import Ui_ManualComponent_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg
from skimage.draw import circle
from skimage.filters import threshold_otsu, rank
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib

class WorkerManualComponent(QtCore.QObject):
    """
    Worker class for the exponential fitting
    Instances of this class can be moved to a thread

    Attributes
    ----------
    img_data: np.ndarray
        the image
    multiplier: float
        multiplication factor for the variance of the seed
    parent: QWidget
        parent widget
    is_abort: bool
        whether the computation was aborted and should be stopped
    """

    #Signal emitted at the end of the computation
    signal_end = QtCore.pyqtSignal(np.ndarray, int)

    number = 1

    def __init__(self, img_data, seed, parent=None):
        super().__init__()
        self.img_data = img_data
        self.seed = seed
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        image = img_as_ubyte(self.img_data * 1.0 / self.img_data.max())
        image = np.reshape(image, (image.shape[0], image.shape[1]) + (-1,), order='F')
        length = image.shape[-1]
        out_image = np.zeros_like(image)
        self.seed = (self.seed[1], self.seed[0])
        for i in range(length):
            QApplication.processEvents()
            current = image[..., i]
            threshold = threshold_otsu(current)
            seg_con = sitk.ConnectedThreshold(sitk.GetImageFromArray(current), seedList=[self.seed], lower=int(threshold), upper=255)
            seg_con_array = sitk.GetArrayFromImage(seg_con)
            out_image[..., i] = seg_con_array
        out_image[out_image == 1] = 255
        out_image = np.reshape(out_image, self.img_data.shape, order='F')
        self.signal_end.emit(out_image, WorkerManualComponent.number)
        if self.is_abort:
            WorkerManualComponent.number += 1


    def abort(self):
        self.is_abort = True

class ManualComponentController:
    """
    Controller handling the ManualComponentView dialog

    Attributes
    ----------
    view: Ui_ManualComponent_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_ManualComponent_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)

        #Events
        self.trigger = Signal()
        self.view.pushButton.clicked.connect(self.trigger.signal.emit)

    def show(self):
        self.dialog.show()
