from src.houghview import Ui_Hough_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg
from skimage.draw import circle
from skimage.filters import threshold_otsu, rank

class WorkerHough(QtCore.QObject):
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

    def __init__(self, img_data, parent=None, min_radius=7, max_radius=20, preview=False):
        super().__init__()
        self.img_data = img_data
        self.min_radius = min_radius
        self.max_radius = max_radius
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
        image8 = np.reshape(image8, (image8.shape[0], image8.shape[1]) + (-1,), order='C')
        length = image8.shape[-1]
        L = np.zeros(shape=(length, 3, 3))
        for i in range(length):
            if self.is_abort:
                break
            QApplication.processEvents()
            image_current = image8[..., i]
            threshold = threshold_otsu(image_current)
            cx, cy, r = seg.detect_circle(image_current, threshold, self.min_radius, self.max_radius)
            L[i, 0] = cx
            L[i, 1] = cy
            L[i, 2] = r
            progress = float(i/length*100)
            self.signal_progress.emit(progress)
        frequent_circle = np.median(L[..., 0], axis=0)
        coordinates = np.delete(np.transpose(L, (0, 2, 1)), -1, axis=2)
        distance_to_frequent_circle = np.linalg.norm(frequent_circle[:-1].T - coordinates, axis=2, ord=2)
        index = np.argmin(distance_to_frequent_circle, axis=1)
        coordinates_circle = np.choose(index, L.T).T

        image_copy = np.zeros_like(self.img_data)
        for i in range(length):
            cx, cy, r = coordinates_circle[i]
            if cx == -1:
                continue
            x, y = circle(cx, cy, r+1, shape=image8[...,0].shape)
            old_index = np.unravel_index(i, self.img_data.shape[2:])
            for j in range(len(x)):
                current_index = (y[j], x[j]) + old_index
                image_copy[current_index] = self.img_data[current_index]

        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(image_copy, WorkerHough.number)
            if not self.is_preview:
                WorkerHough.number += 1


    def abort(self):
        self.is_abort = True

class HoughController:
    """
    Controller handling the HoughView dialog

    Attributes
    ----------
    view: Ui_Hough_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_Hough_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)
        self.view.pushButton.setFixedWidth(20)
        self.view.pushButton_2.setFixedWidth(20)


        #Tooltips
        t1 = self.view.pushButton.toolTip()
        t2 = self.view.pushButton_2.toolTip()
        self.view.pushButton.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t1)
        self.view.pushButton_2.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t2)
        #Reset tooltips to avoid overlap of events
        self.view.pushButton.setToolTip("")
        self.view.pushButton_2.setToolTip("")

        #Events
        self.trigger = Signal()

        self.view.buttonBox.accepted.connect(self.update_parameters)


    def update_parameters(self, preview=False):
        """
        Gets the values in the GUI and updates the attributes
        """
        self.min_radius = self.view.lineEdit.text()
        self.max_radius = self.view.lineEdit_2.text()
        if not preview:
            self.trigger.signal.emit()

    def show(self):
        self.dialog.show()
