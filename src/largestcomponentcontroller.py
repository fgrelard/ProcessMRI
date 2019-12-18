from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg

class WorkerLargestComponent(QtCore.QObject):
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

    #PyQt5 signals
    #Signal emitted at the start of the computation
    signal_start = QtCore.pyqtSignal()

    #Signal emitted at the end of the computation
    signal_end = QtCore.pyqtSignal(np.ndarray, int)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, parent=None, preview=False):
        super().__init__()
        self.img_data = img_data
        self.is_preview = preview
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to cavity.exponentialfit_image
        """
        self.signal_start.emit()
        image8 = img_as_ubyte(self.img_data * 1.0 / self.img_data.max())
        image_copy = self.img_data.copy()
        length = image8.shape[-1]
        for i in range(length):
            if self.is_abort:
                break
            QApplication.processEvents()
            binarized = seg.binarize(image8[..., i])
            grain = seg.largest_connected_component(binarized)
            cond = np.where(grain == 0) + (i, )
            image_copy[cond] = 0
            progress = float(i/length*100)
            self.signal_progress.emit(progress)
        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(image_copy, WorkerLargestComponent.number)
            if not self.is_preview:
                WorkerLargestComponent.number += 1


    def abort(self):
        self.is_abort = True
