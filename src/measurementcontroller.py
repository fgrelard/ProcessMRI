from src.measurementview import Ui_Measurement_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.measurements as measurements
from skimage.draw import circle
from skimage.filters import threshold_otsu, rank
import src.imageio as io

class WorkerMeasurement(QtCore.QObject):
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
    signal_end = QtCore.pyqtSignal(list, list, np.ndarray)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, images, parent=None, metadata=None, names=[], slice_range=-1, ):
        super().__init__()
        self.images = images
        self.names = names
        self.slice_range = slice_range
        self.parent = parent
        self.metadata = metadata
        self.is_abort = False


    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to cavity.exponentialfit_image
        """
        self.signal_start.emit()
        out = np.zeros(shape=(len(self.names), 5))
        i = 0
        units = []
        for name in self.names:
            image = self.images[name]
            if name in self.metadata:
                metadata = self.metadata[name]
            else:
                metadata = None
            res, unit = io.extract_resolution(metadata)
            units.append(unit)

            image = np.reshape(image, (image.shape[0], image.shape[1]) + (-1,), order='C')
            if isinstance(self.slice_range, (np.ndarray, list)):
                self.slice_range = [x for x in self.slice_range if x in range(image.shape[-1])]
                image = image[..., self.slice_range]
            area_pix = measurements.area_pixels(image)
            area_unit = measurements.area_unit(image, res)
            average = measurements.average_value(image)
            min = measurements.min_value(image)
            max = measurements.max_value(image)
            out[i, 0] = area_pix
            out[i, 1] = area_unit
            out[i, 2] = average
            out[i, 3] = min
            out[i, 4] = max
            i +=1
        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(self.names, units, out)


    def abort(self):
        self.is_abort = True

class MeasurementController:
    """
    Controller handling the MeasurementView dialog

    Attributes
    ----------
    view: Ui_Measurement_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_Measurement_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)
        self.view.pushButton.setFixedWidth(20)


        #Tooltips
        t1 = self.view.pushButton.toolTip()
        self.view.pushButton.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t1)
        #Reset tooltips to avoid overlap of events
        self.view.pushButton.setToolTip("")



        #Events
        self.trigger = Signal()
        self.view.buttonBox.accepted.connect(self.update_parameters)

    def add_items(self, items):
        self.view.comboBox.clear()
        self.view.comboBox.addItem("All")
        self.view.comboBox.addItems(items)


    def update_parameters(self, preview=False):
        """
        Gets the values in the GUI and updates the attributes
        """
        self.image = self.view.comboBox.currentText()
        self.slice_range = self.view.lineEdit.text()
        self.trigger.signal.emit()


    def show(self, items=[]):
        self.add_items(items)
        self.dialog.show()
