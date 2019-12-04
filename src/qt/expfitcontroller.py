from src.qt.expfitview import Ui_ExpFit_View
from PyQt5.QtWidgets import QDialog, QMainWindow, QToolTip
from src.qt.signal import Signal
from PyQt5 import QtCore

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
