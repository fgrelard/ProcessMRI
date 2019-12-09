from src.tpcview import Ui_TPC_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.temporalphasecorrection as tpc
import src.compleximage as ci
import cmath

class WorkerTPC(QtCore.QObject):
    """
    Worker class for TPC
    Instances of this class can be moved to a thread

    Attributes
    ----------
    img_data: np.ndarray
        the image
    echotime: np.ndarray
        echotimes
    order: int
        polynomial order
    threshold: int
        threshold for exponential fitting
    is_abort: bool
        whether the computation was aborted and should be stopped
    """

    #PyQt5 signals
    #Signal emitted at the start of the computation
    signal_start = QtCore.pyqtSignal()

    #Signal emitted at the end of the computation
    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, int)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, echotime, order, threshold):
        super().__init__()
        self.img_data = img_data
        self.echotime = np.array(echotime).tolist()
        self.polynomial_order = order
        self.threshold = threshold
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to tpc.correct_phase_temporally
        """
        self.signal_start.emit()
        out_img_data = np.zeros(shape=(self.img_data.shape[:-1]+ (self.img_data.shape[-1]//2, )), dtype=complex)
        ri = self.img_data.shape[-1]
        complex_img_data = ci.to_complex(self.img_data)
        even_echotime = self.echotime[:ri//2:2]
        odd_echotime = self.echotime[1:ri//2:2]

        magnitude_img = ci.complex_to_magnitude(complex_img_data)

        #Separating even and odd echoes
        even_complex_img = complex_img_data[..., ::2]
        odd_complex_img = complex_img_data[..., 1::2]
        phase_image = ci.complex_to_phase(complex_img_data)
        phases_unwrapped = tpc.unwrap_phases(phase_image)

        #Iterating over the even and odd slices
        length = even_complex_img.size
        for index in np.ndindex(even_complex_img.shape[:-1]):
            if self.is_abort:
                break
            QApplication.processEvents()
            phase_unwrapped_even = phases_unwrapped[index + (slice(None, None, 2),)]
            phase_unwrapped_odd = phases_unwrapped[index + (slice(1, None, 2),)]
            tpc_even = tpc.correct_phase_1d(even_echotime, even_complex_img[index], self.polynomial_order)
            tpc_even = tpc.correct_phase_1d(even_echotime, tpc_even, self.polynomial_order)
            tpc_odd = tpc.correct_phase_1d(odd_echotime, odd_complex_img[index], self.polynomial_order)
            tpc_odd = tpc.correct_phase_1d(odd_echotime, tpc_odd, self.polynomial_order)
            for k in range(out_img_data.shape[-1]):
                pointwise_index = index + (k, )
                value = magnitude_img[pointwise_index]
                if value < self.threshold:
                    out_img_data[pointwise_index] = complex_img_data[pointwise_index]
                elif k % 2 == 0:
                    out_img_data[pointwise_index] = tpc_even[k//2]
                else:
                    out_img_data[pointwise_index] = tpc_odd[k//2]
            i = np.ravel_multi_index(index, even_complex_img.shape[:-1])
            progress = float(i / length * 100)
            self.signal_progress.emit(progress)
        if not self.is_abort:
            real = out_img_data.real
            imag = out_img_data.imag
            magnitude = ci.complex_to_magnitude(out_img_data)
            phase = ci.complex_to_phase(out_img_data)
            self.signal_end.emit(real, imag, magnitude, phase, WorkerTPC.number)
            WorkerTPC.number += 1

    def abort(self):
        self.is_abort = True

class TPCController:
    """
    Controller handling the TPCView dialog

    Attributes
    ----------
    view: Ui_TPC_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_TPC_View()
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


    def update_parameters(self):
        """
        Gets the values in the GUI and updates the attributes
        """
        self.polynomial_order = self.view.lineEdit.text()
        self.threshold = self.view.lineEdit_2.text()
        self.trigger.signal.emit()


    def show(self):
        self.dialog.show()
