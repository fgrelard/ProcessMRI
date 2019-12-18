from src.houghview import Ui_Hough_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg

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
        image_copy = self.img_data.copy()
        length = image8.shape[-1]
        L = np.zeros(shape=(length, 3))
        for i in range(length):
            if self.is_abort:
                break
            QApplication.processEvents()
            image_current = image8[..., i]
            threshold = threshold_otsu(image_current)
            cavity = seg.detect_circle(image_current, threshold, self.min_radius, self.max_radius)
            L[i, 0] = cx
            L[i, 1] = cy
            L[i, 2] = r
            progress = float(i/length*100)
            self.signal_progress.emit(progress)
        median_circle = np.median(L, axis=0)
        np.linalg.norm(median_circle.T - L, axis=1, ord=2)
        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(median_circle, WorkerHough.number)
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
        value = self.view.horizontalSlider.value()
        self.min_radius = self.slidervalue_to_multvalue(value)
        self.max_radius = self.view.lineEdit.text()
        if not preview:
            self.trigger.signal.emit()

    def show(self):
        self.dialog.show()
