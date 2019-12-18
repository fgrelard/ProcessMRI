from src.cavityview import Ui_Cavity_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QSlider
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.segmentation as seg

class WorkerCavity(QtCore.QObject):
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

    def __init__(self, img_data, parent=None, multiplier=2.5, start=1, end=0, preview=False):
        super().__init__()
        self.img_data = img_data
        self.multiplier = multiplier
        self.start_slice = start
        self.end_slice = end
        self.is_preview = preview
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to cavity.exponentialfit_image
        """
        self.signal_start.emit()
        depth = self.img_data.shape[-1]
        if self.end_slice > depth:
            self.end_slice = depth
        if self.start_slice > depth:
            self.start_slice = 0
        if self.end_slice <= self.start_slice:
            self.start_slice = 0
            self.end_slice = depth
        image8 = img_as_ubyte(self.img_data * 1.0 / self.img_data.max())
        image8 = image8[..., self.start_slice:self.end_slice]
        image_copy = self.img_data.copy()[..., self.start_slice:self.end_slice]
        length = image8.shape[-1]
        for i in range(length):
            if self.is_abort:
                break
            QApplication.processEvents()
            cavity = seg.detect_cavity(image8[ ..., i], multiplier=self.multiplier)
            cond = np.where(cavity == 0) + (i, )
            image_copy[cond] = 0
            progress = float(i/depth*100)
            self.signal_progress.emit(progress)
        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(image_copy, WorkerCavity.number)
            if not self.is_preview:
                WorkerCavity.number += 1


    def abort(self):
        self.is_abort = True

class CavityController:
    """
    Controller handling the CavityView dialog

    Attributes
    ----------
    view: Ui_Cavity_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_Cavity_View()
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
        self.view.horizontalSlider.mouseMoveEvent = self.slider_event
        self.view.buttonBox.accepted.connect(self.update_parameters)


    def update_tooltip(self):
        value = self.view.horizontalSlider.value()
        value = self.slidervalue_to_multvalue(value)
        self.view.horizontalSlider.setToolTip(str(value))

    def slider_event(self, event):
        self.update_tooltip()
        QToolTip.showText(event.globalPos(), self.view.horizontalSlider.toolTip())
        QSlider.mouseMoveEvent(self.view.horizontalSlider, event)

    def slidervalue_to_multvalue(self, value):
        return float(value/10) + 1

    def update_parameters(self, preview=False):
        """
        Gets the values in the GUI and updates the attributes
        """
        value = self.view.horizontalSlider.value()
        self.multiplier = self.slidervalue_to_multvalue(value)
        self.start_slice = self.view.lineEdit.text()
        self.end_slice = self.view.lineEdit_2.text()
        if not preview:
            self.trigger.signal.emit()

    def show(self):
        self.dialog.show()
