# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/manualsegmentationview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ManualSegmentation_View(object):
    def setupUi(self, ManualSegmentation_View):
        ManualSegmentation_View.setObjectName("ManualSegmentation_View")
        ManualSegmentation_View.resize(479, 310)
        self.buttonBox = QtWidgets.QDialogButtonBox(ManualSegmentation_View)
        self.buttonBox.setGeometry(QtCore.QRect(200, 240, 251, 32))
        self.buttonBox.setToolTipDuration(-1)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(ManualSegmentation_View)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 30, 421, 194))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
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
        self.gridLayout.addWidget(self.textEdit, 0, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(ManualSegmentation_View)
        self.pushButton.setGeometry(QtCore.QRect(30, 240, 91, 23))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(ManualSegmentation_View)
        self.buttonBox.accepted.connect(ManualSegmentation_View.accept)
        self.buttonBox.rejected.connect(ManualSegmentation_View.reject)
        QtCore.QMetaObject.connectSlotsByName(ManualSegmentation_View)

    def retranslateUi(self, ManualSegmentation_View):
        _translate = QtCore.QCoreApplication.translate
        ManualSegmentation_View.setWindowTitle(_translate("ManualSegmentation_View", "Manual segmentation"))
        self.textEdit.setHtml(_translate("ManualSegmentation_View", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Manual segmentation </span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Press the &quot;Start&quot; button to start the manual segmentation. Next, left-click on the pixels you want to add to the segmentation. You may remove pixels by right-clicking on the desired pixels. You can switch slices while on this mode. </p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Press the &quot;OK button&quot; to terminate the segmentation process. A binary image will be generated.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("ManualSegmentation_View", "Start"))


