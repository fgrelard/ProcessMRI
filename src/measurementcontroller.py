from src.measurementview import Ui_Measurement_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication, QTableWidget, QTableWidgetItem
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
from skimage.util import img_as_ubyte
import src.measurements as measurements
from skimage.draw import circle
from skimage.filters import threshold_otsu, rank

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
    signal_end = QtCore.pyqtSignal(float, float, float, float, float)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, parent=None, slice_range=-1, resolution=(1,1,1)):
        super().__init__()
        self.img_data = img_data
        self.slice_range = slice_range
        self.resolution = resolution
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to cavity.exponentialfit_image
        """
        self.signal_start.emit()
        area_pix = measurements.area_pixels(self.img_data)
        area_unit = measurements.area_unit(self.img_data, self.resolution)
        average = measurements.average_value(self.img_data)
        min = measurements.min_value(self.img_data)
        max = measurements.max_value(self.img_data)

        if not self.is_abort:
            #Send images as a signal
            self.signal_end.emit(area_pix, area_unit, average, min, max)


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
