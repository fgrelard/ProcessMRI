from src.qt.nlmeansview import Ui_NLMeans_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip, QApplication
from src.qt.signal import Signal
from PyQt5 import QtCore
import numpy as np
import src.exponentialfit as expfit


class WorkerNLMeans(QtCore.QObject):

    signal_start = QtCore.pyqtSignal()
    signal_end = QtCore.pyqtSignal(np.ndarray, int)
    signal_progress = QtCore.pyqtSignal(int)
    number = 1

    def __init__(self, img_data, patch_size, patch_distance, noise_spread):
        super().__init__()
        self.img_data = img_data
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.noise_spread = noise_spread
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        self.signal_start.emit()
        denoised = np.zeros_like(self.img_data)
        dim = len(self.img_data.shape)
        if dim > 3:
            length = self.img_data.shape[-1]
            for i in range(self.img_data.shape[-1]):
                if self.is_abort:
                    break
                QApplication.processEvents()
                image3D = expfit.denoise_2_3D(self.img_data[...,i], self.patch_size, self.patch_distance, self.noise_spread)
                denoised[..., i] = image3D
                progress = float(i / length * 100)
                self.signal_progress.emit(progress)
        else:
            denoised = expfit.denoise_2_3D(self.img_data, self.patch_size, self.patch_distance, self.noise_spread)
        if not self.is_abort:
            self.signal_end.emit(denoised, WorkerNLMeans.number)
            WorkerNLMeans.number += 1


    def abort(self):
        self.is_abort = True

class NLMeansController:
    def __init__(self, app):
        self.dialog = QDialog()
        self.dialog.parent = app

        #Move dialog
        app.move_dialog(self.dialog)

        #Init ui
        self.view = Ui_NLMeans_View()
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
        self.view.buttonBox.accepted.connect(self.update_parameters)


    def update_parameters(self):
        self.patch_size = self.view.lineEdit.text()
        self.patch_distance = self.view.lineEdit_2.text()
        self.noise_spread = self.view.lineEdit_3.text()
        self.trigger.signal.emit()


    def show(self):
        self.dialog.show()
