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

    def __init__(self, img_data, seed, multiplier=1.0, parent=None):
        super().__init__()
        self.img_data = img_data
        self.seed = seed
        self.multiplier = multiplier
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        image = img_as_ubyte(self.img_data * 1.0 / self.img_data.max())
        image = np.reshape(image, (image.shape[0], image.shape[1]) + (-1,), order='F')
        length = image.shape[-1]
        seg_image = np.zeros_like(image)
        self.seed = (self.seed[1], self.seed[0])
        if not (0 <= self.seed[0] < image.shape[1] and 0 <= self.seed[1] < image.shape[0]):
            return
        for i in range(length):
            QApplication.processEvents()
            current = image[..., i]
            try:
                threshold = threshold_otsu(current)
                threshold = max(1, min(int(threshold*self.multiplier), 255))
            except:
                threshold = 50
            seg_con = sitk.ConnectedThreshold(sitk.GetImageFromArray(current), seedList=[self.seed], lower=int(threshold), upper=255)
            seg_con_array = sitk.GetArrayFromImage(seg_con)
            seg_image[..., i] = seg_con_array
        seg_image = np.reshape(seg_image, self.img_data.shape, order='F')
        out_image = self.img_data.copy()
        out_image[seg_image != 1] = 0
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
        self.view.pushButton_2.setFixedWidth(20)


        t1 = self.view.pushButton_2.toolTip()
        self.view.pushButton_2.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t1)
        #Reset tooltips to avoid overlap of events
        self.view.pushButton_2.setToolTip("")

        #Events
        self.trigger = Signal()
        self.view.pushButton.clicked.connect(self.update_parameters)
        self.view.horizontalSlider.mouseMoveEvent = self.slider_event

    def update_parameters(self):
        value = self.view.horizontalSlider.value()
        self.multiplier = self.slidervalue_to_multvalue(value)
        self.trigger.signal.emit()

    def show(self):
        self.dialog.show()

    def slider_event(self, event):
        self.update_tooltip()
        QToolTip.showText(event.globalPos(), self.view.horizontalSlider.toolTip())
        QSlider.mouseMoveEvent(self.view.horizontalSlider, event)

    def update_tooltip(self):
        value = self.view.horizontalSlider.value()
        value = self.slidervalue_to_multvalue(value)
        self.view.horizontalSlider.setToolTip(str(value))

    def slidervalue_to_multvalue(self, value):
        return float(value/10)
