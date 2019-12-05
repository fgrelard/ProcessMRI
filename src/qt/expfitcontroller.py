from src.qt.expfitview import Ui_ExpFit_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.qt.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.exponentialfit as expfit


class WorkerExpFit(QtCore.QObject):

    signal_start = QtCore.pyqtSignal()
    signal_end = QtCore.pyqtSignal(np.ndarray, np.ndarray, int)
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, echotime, parent=None, threshold=None, lreg=True, n=1):
        super().__init__()
        self.img_data = img_data
        self.echotime = echotime
        self.threshold = threshold
        self.lreg = lreg
        self.n = n
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        self.signal_start.emit()
        echotime = self.echotime
        image = self.img_data
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
            QApplication.processEvents()
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
            self.signal_end.emit(density_data, t2_data, WorkerExpFit.number)
            WorkerExpFit.number += 1


    def abort(self):
        self.is_abort = True

class ExpFitController:
    def __init__(self, app):
        self.dialog = QDialog()
        self.dialog.parent = app

        #Move dialog
        app.move_dialog(self.dialog)

        #Init ui
        self.view = Ui_ExpFit_View()
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
        self.fit_method = self.view.comboBox.currentText()
        self.threshold = self.view.lineEdit.text()
        self.trigger.signal.emit()


    def show(self):
        self.dialog.show()
