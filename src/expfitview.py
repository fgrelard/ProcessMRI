# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/expfitview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ExpFit_View(object):
    def setupUi(self, ExpFit_View):
        ExpFit_View.setObjectName("ExpFit_View")
        ExpFit_View.resize(449, 286)
        self.buttonBox = QtWidgets.QDialogButtonBox(ExpFit_View)
        self.buttonBox.setGeometry(QtCore.QRect(70, 230, 341, 32))
        self.buttonBox.setToolTipDuration(-1)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(ExpFit_View)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 30, 410, 186))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 5, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setMouseTracking(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 5, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 4, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 3, 1, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setSizeIncrement(QtCore.QSize(0, 0))
        self.textEdit.setBaseSize(QtCore.QSize(0, 0))
        self.textEdit.setAutoFillBackground(True)
        self.textEdit.setStyleSheet("background: rgba(0,0,0,0%)")
        self.textEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 0, 0, 1, 3)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 4, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setMouseTracking(True)
        self.pushButton_2.setToolTipDuration(-1)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 3, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMouseTracking(True)
        self.pushButton.setToolTipDuration(-1)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 4, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 6, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 6, 1, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_4.setEnabled(False)
        self.pushButton_4.setMouseTracking(True)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 6, 2, 1, 1)

        self.retranslateUi(ExpFit_View)
        self.buttonBox.accepted.connect(ExpFit_View.accept)
        self.buttonBox.rejected.connect(ExpFit_View.reject)
        QtCore.QMetaObject.connectSlotsByName(ExpFit_View)

    def retranslateUi(self, ExpFit_View):
        _translate = QtCore.QCoreApplication.translate
        ExpFit_View.setWindowTitle(_translate("ExpFit_View", "Exponential fit"))
        self.lineEdit_2.setText(_translate("ExpFit_View", "1.0"))
        self.pushButton_3.setToolTip(_translate("ExpFit_View", "Threshold on normalized residual errors to discard wrong fits. \n"
"Pixels with residual errors above this threshold are discarded. \n"
"Accepted values are in the range [0, 1]. A value of 1 retains all the fitted values."))
        self.pushButton_3.setText(_translate("ExpFit_View", "?"))
        self.label_2.setText(_translate("ExpFit_View", "Threshold"))
        self.label.setText(_translate("ExpFit_View", "Fit method"))
        self.comboBox.setItemText(0, _translate("ExpFit_View", "Linear regression"))
        self.comboBox.setItemText(1, _translate("ExpFit_View", "Linear regression bi-exponential"))
        self.comboBox.setItemText(2, _translate("ExpFit_View", "Piecewise linear regression"))
        self.comboBox.setItemText(3, _translate("ExpFit_View", "NNLS mono-exponential"))
        self.comboBox.setItemText(4, _translate("ExpFit_View", "NNLS bi-exponential"))
        self.comboBox.setItemText(5, _translate("ExpFit_View", "NNLS tri-exponential"))
        self.textEdit.setHtml(_translate("ExpFit_View", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Exponential</span><span style=\" font-size:10pt;\"> </span><span style=\" font-size:10pt; font-weight:600;\">fit</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Fit a n-exponential function on multiple echo data.</p></body></html>"))
        self.lineEdit.setText(_translate("ExpFit_View", "Auto"))
        self.pushButton_2.setToolTip(_translate("ExpFit_View", "Fit method: linear regression on the log of the data, \n"
"or non-negative least squares fitting of n-exponential"))
        self.pushButton_2.setText(_translate("ExpFit_View", "?"))
        self.pushButton.setToolTip(_translate("ExpFit_View", "Threshold on pixel values to discard low SNR pixels from the fitting. \n"
"Default: auto threshold based on gaussian mixture on the histogram values. \n"
"Set the value to 0 for no threshold"))
        self.pushButton.setText(_translate("ExpFit_View", "?"))
        self.label_3.setText(_translate("ExpFit_View", "Threshold residuals"))
        self.label_4.setText(_translate("ExpFit_View", "Threshold exp. factor"))
        self.lineEdit_3.setText(_translate("ExpFit_View", "0.05"))
        self.pushButton_4.setToolTip(_translate("ExpFit_View", "Threshold on exponential factor to discard wrong fits. \n"
"When exp factor is very small, the fit is a line. \n"
"Pixels with exponential factor below this threshold are discarded. \n"
"Accepted values are in the range [0, 1].\n"
" Default value=0.05. A value of 0 retains all the fitted values."))
        self.pushButton_4.setText(_translate("ExpFit_View", "?"))


