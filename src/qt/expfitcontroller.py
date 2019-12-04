from src.qt.expfitview import Ui_ExpFit_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip
from src.qt.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.exponentialfit as expfit


class WorkerExpFit(QtCore.QObject):

    signal_start = QtCore.pyqtSignal()
    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray, int)
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, maincontroller, parent=None, threshold=None, lreg=True, n=1):
        super().__init__()
        self.maincontroller = maincontroller
        self.threshold = threshold
        self.lreg = lreg
        self.n = n
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        self.signal_start.emit()
        echotime = self.maincontroller.echotime
        image = self.maincontroller.img_data
        threshold = self.threshold
        lreg = self.lreg
        n = self.n
        density_data = np.zeros(shape=image.shape[:-1])
        t2_data = np.zeros(shape=image.shape[:-1])

    #Auto threshold with mixture of gaussian (EM alg.)
        if threshold is None:
            threshold = expfit.auto_threshold_gmm(np.expand_dims(image[...,0].ravel(), 1), 3)

        length = density_data.size
        for i in np.ndindex(density_data.shape):
            if self.is_abort:
                break
            self.maincontroller.app.processEvents()
            pixel_values = image[i + (slice(None),)]
            if pixel_values[0] > threshold:
                p0 = expfit.n_to_p0(n, pixel_values[0])
                fit = expfit.fit_exponential(echotime, pixel_values, p0, lreg)
                density_value = expfit.density(fit)
                t2_value = expfit.t2_star(fit, echotime[0])

                density_data[i] = density_value
                t2_data[i] = t2_value
            else:
                density_data[i] = pixel_values[0]
                t2_data[i] = 0
            index = np.ravel_multi_index(i, density_data.shape)
            progress = float(index/length*100)
            self.signal_progress.emit(progress)
        if not self.is_abort:
            self.signal_end.emit(density_data, t2_data, self.number)
            self.number += 1


    def abort(self):
        self.is_abort = True

class ExpFitController:
    def __init__(self, app):
        self.dialog = QDialog()
        self.dialog.parent = app

        #Move dialog
        app.move_dialog(self.dialog)

        #Init ui
        self.expfitview = Ui_ExpFit_View()
        self.expfitview.setupUi(self.dialog)
        self.expfitview.retranslateUi(self.dialog)
        self.expfitview.pushButton.setFixedWidth(20)
        self.expfitview.pushButton_2.setFixedWidth(20)

        #Tooltips
        self.expfitview.pushButton.enterEvent = lambda event : QToolTip.showText(event.globalPos(), self.expfitview.pushButton.toolTip())
        self.expfitview.pushButton_2.enterEvent = lambda event : QToolTip.showText(event.globalPos(), self.expfitview.pushButton_2.toolTip())
        #Reset tooltips to avoid overlap of events
        self.expfitview.pushButton.setToolTip("")
        self.expfitview.pushButton_2.setToolTip("")

        #Events
        self.signal = Signal()
        self.expfitview.buttonBox.accepted.connect(self.exp_fit_parameters)


    def exp_fit_parameters(self):
        self.fit_method = self.expfitview.comboBox.currentText()
        self.threshold = self.expfitview.lineEdit.text()
        self.signal.compute_signal.emit()


    def show(self):
        self.dialog.show()
