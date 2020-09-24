from src.expfitview import Ui_ExpFit_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.exponentialfit as expfit
import matplotlib.pyplot as plt


class WorkerExpFit(QtCore.QObject):
    """
    Worker class for the exponential fitting
    Instances of this class can be moved to a thread

    Attributes
    ----------
    img_data: np.ndarray
        the image
    echotime: np.ndarray
        echotimes
    parent: QWidget
        parent widget
    threshold: int
        threshold for exponential fitting
    lreg: bool
        whether to use linear regression or nnls
    n: int
        number of exponentials
    is_abort: bool
        whether the computation was aborted and should be stopped
    """

    #PyQt5 signals
    #Signal emitted at the start of the computation
    signal_start = QtCore.pyqtSignal()

    #Signal emitted at the end of the computation
    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray, int)

    #Signal emitted during the computation, to keep
    #track of its progress
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, echotime, parent=None, threshold=None, lreg=True, biexp=False, piecewise_lreg=False, n=1, threshold_error=1.0, threshold_expfactor=0):
        super().__init__()
        self.img_data = img_data
        self.echotime = echotime
        self.threshold = threshold
        self.lreg = lreg
        self.biexp = biexp
        self.piecewise_lreg = piecewise_lreg
        self.n = n
        self.is_abort = False
        self.threshold_error = threshold_error
        self.threshold_expfactor = threshold_expfactor

    @QtCore.pyqtSlot()
    def work(self):
        """
        Computation of exponential fitting

        Analogous to expfit.exponentialfit_image
        """
        self.signal_start.emit()
        echotime = self.echotime
        image = self.img_data
        threshold = self.threshold
        lreg = self.lreg
        biexp = self.biexp
        piecewise_lreg = self.piecewise_lreg
        density_data = np.zeros(shape=image.shape[:-1])
        t2_data = np.zeros(shape=image.shape[:-1])
        fit_data = np.zeros(shape=image.shape[:-1] + (2*self.n+1, ))
        residual_data = np.zeros(shape=image.shape[:-1])

        #Auto threshold with mixture of gaussian (EM alg.)
        if threshold is None:
            threshold = expfit.auto_threshold_gmm(np.expand_dims(image[...,0].ravel(), 1), 3)

        length = density_data.size

        for i in np.ndindex(density_data.shape):
            if self.is_abort:
                break
            QApplication.processEvents()
            pixel_values = image[i + (slice(None),)]
            if pixel_values[0] > threshold:
                p0 = expfit.n_to_p0(self.n, pixel_values[0])
                fit, residual = expfit.fit_exponential(echotime, pixel_values, p0, lreg, biexp, piecewise_lreg)
                fit_data[i] = fit
                residual_data[i] = residual

                density_value = expfit.density(fit)
                t2_value = expfit.t2_star(fit, echotime[0])

                if np.sum(fit[1::2]) > self.threshold_expfactor and \
                   residual < self.threshold_error:
                    density_data[i] = density_value
                    t2_data[i] = t2_value
                else:
                    density_data[i] = pixel_values[0]
                    t2_data[i] = 0
            else:
                density_data[i] = pixel_values[0]
                t2_data[i] = 0
            index = np.ravel_multi_index(i, density_data.shape)
            progress = float(index/length*100)
            self.signal_progress.emit(progress)
        if not self.is_abort:
            #Send images as a signal
            density_data = np.nan_to_num(density_data)
            t2_data = np.nan_to_num(t2_data)
            density_data = np.concatenate((density_data[...,None], image, fit_data, residual_data[..., None]), axis=-1)
            t2_data = np.concatenate((t2_data[..., None], image, fit_data, residual_data[..., None]), axis=-1)
            self.signal_end.emit(density_data, t2_data, WorkerExpFit.number)
            WorkerExpFit.number += 1


    def abort(self):
        self.is_abort = True

class ExpFitController:
    """
    Controller handling the ExpFitView dialog

    Attributes
    ----------
    view: Ui_ExpFit_View
        the view
    trigger: Signal
        signal raised when clicking on the "OK" button
    """
    def __init__(self, parent):
        self.dialog = QDialog(parent)

        #Init ui
        self.view = Ui_ExpFit_View()
        self.view.setupUi(self.dialog)
        self.view.retranslateUi(self.dialog)
        self.view.pushButton.setFixedWidth(20)
        self.view.pushButton_2.setFixedWidth(20)
        self.view.pushButton_3.setFixedWidth(20)
        self.view.pushButton_4.setFixedWidth(20)



        #Tooltips
        t1 = self.view.pushButton.toolTip()
        t2 = self.view.pushButton_2.toolTip()
        t3 = self.view.pushButton_3.toolTip()
        t4 = self.view.pushButton_4.toolTip()
        self.view.pushButton.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t1)
        self.view.pushButton_2.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t2)
        self.view.pushButton_3.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t3)
        self.view.pushButton_4.enterEvent = lambda event : QToolTip.showText(event.globalPos(), t4)
        #Reset tooltips to avoid overlap of events
        self.view.pushButton.setToolTip("")
        self.view.pushButton_2.setToolTip("")
        self.view.pushButton_3.setToolTip("")
        self.view.pushButton_4.setToolTip("")

        #Events
        self.trigger = Signal()
        self.view.buttonBox.accepted.connect(self.update_parameters)

        self.fit_method = self.view.comboBox.currentText()
        self.threshold = self.view.lineEdit.text()
        self.threshold_error = self.view.lineEdit_2.text()
        self.threshold_expfactor = self.view.lineEdit_3.text()


    def update_parameters(self):
        """
        Gets the values in the GUI and updates the attributes
        """
        self.fit_method = self.view.comboBox.currentText()
        self.threshold = self.view.lineEdit.text()
        self.threshold_error = self.view.lineEdit_2.text()
        self.threshold_expfactor = self.view.lineEdit_3.text()
        self.trigger.signal.emit()


    def show(self):
        self.dialog.show()
