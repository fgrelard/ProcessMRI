# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/manualcomponentview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ManualComponent_View(object):
    def setupUi(self, ManualComponent_View):
        ManualComponent_View.setObjectName("ManualComponent_View")
        ManualComponent_View.resize(479, 329)
        self.buttonBox = QtWidgets.QDialogButtonBox(ManualComponent_View)
        self.buttonBox.setGeometry(QtCore.QRect(200, 270, 251, 32))
        self.buttonBox.setToolTipDuration(-1)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(ManualComponent_View)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 30, 421, 223))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 1, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.gridLayoutWidget)
        self.horizontalSlider.setMouseTracking(True)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setProperty("value", 10)
        self.horizontalSlider.setTracking(False)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 2, 2, 1, 1)
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
        self.gridLayout.addWidget(self.pushButton_2, 2, 3, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setSizeIncrement(QtCore.QSize(0, 0))
        self.textEdit.setBaseSize(QtCore.QSize(0, 0))
        self.textEdit.setAutoFillBackground(True)
        self.textEdit.setStyleSheet("background: rgba(0,0,0,0%)")
        self.textEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 1, 1, 1, 3)
        self.pushButton = QtWidgets.QPushButton(ManualComponent_View)
        self.pushButton.setGeometry(QtCore.QRect(30, 270, 91, 23))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(ManualComponent_View)
        self.buttonBox.accepted.connect(ManualComponent_View.accept)
        self.buttonBox.rejected.connect(ManualComponent_View.reject)
        QtCore.QMetaObject.connectSlotsByName(ManualComponent_View)

    def retranslateUi(self, ManualComponent_View):
        _translate = QtCore.QCoreApplication.translate
        ManualComponent_View.setWindowTitle(_translate("ManualComponent_View", "Manual component extraction"))
        self.label.setText(_translate("ManualComponent_View", "Multiplier"))
        self.pushButton_2.setToolTip(_translate("ManualComponent_View", "Multiplication factor for the threshold for the region growing."))
        self.pushButton_2.setText(_translate("ManualComponent_View", "?"))
        self.textEdit.setHtml(_translate("ManualComponent_View", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Semi-manual component extraction</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Extracts a component in the image from a selected seed point, by region growing.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Press the &quot;Start&quot; button to start the process. Next, left-click on the desired component.  </p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Validate the segmentation by pressing &quot;OK&quot;, or generate another segmentation by selecting another seed.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("ManualComponent_View", "Start"))


