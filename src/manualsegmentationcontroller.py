from src.manualsegmentationview import Ui_ManualSegmentation_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg
from skimage.draw import circle
from skimage.filters import threshold_otsu, rank

class WorkerManualSegmentation(QtCore.QObject):
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

    def __init__(self, img_data, parent=None):
        super().__init__()
        self.img_data = img_data
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        while True:
            QApplication.processEvents()
            if self.is_abort:
                break
        image = self.img_data
        maximum = np.amax(image)
        cond = np.where(image == np.amax(image))
        print(cond)
        segmentation = np.zeros_like(image)
        segmentation[cond] = image[cond]
        self.signal_end.emit(segmentation, WorkerManualSegmentation.number)
        WorkerManualSegmentation.number += 1


    def abort(self):
        self.is_abort = True

class ManualSegmentationController:
    """
    Controller handling the ManualSegmentationView dialog

    Attributes
    ----------
    view: Ui_ManualSegmentation_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_ManualSegmentation_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)

        #Events
        self.trigger = Signal()
        self.view.pushButton.clicked.connect(self.trigger.signal.emit)

    def show(self):
        self.dialog.show()
