# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/cavityview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Cavity_View(object):
    def setupUi(self, Cavity_View):
        Cavity_View.setObjectName("Cavity_View")
        Cavity_View.resize(443, 386)
        self.buttonBox = QtWidgets.QDialogButtonBox(Cavity_View)
        self.buttonBox.setGeometry(QtCore.QRect(90, 330, 341, 32))
        self.buttonBox.setToolTipDuration(-1)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(Cavity_View)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 20, 421, 271))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.horizontalSlider_2.setMouseTracking(True)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(100)
        self.horizontalSlider_2.setSingleStep(1)
        self.horizontalSlider_2.setProperty("value", 25)
        self.horizontalSlider_2.setTracking(False)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setInvertedAppearance(False)
        self.horizontalSlider_2.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout.addWidget(self.horizontalSlider_2, 6, 2, 1, 1)
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
        self.gridLayout.addWidget(self.pushButton_2, 6, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 6, 1, 1, 1)
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
        self.gridLayout.addWidget(self.textEdit, 0, 0, 1, 4)
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
        self.gridLayout.addWidget(self.pushButton, 2, 3, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.gridLayoutWidget)
        self.horizontalSlider.setMouseTracking(True)
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setMaximum(25)
        self.horizontalSlider.setSingleStep(2)
        self.horizontalSlider.setPageStep(4)
        self.horizontalSlider.setProperty("value", 5)
        self.horizontalSlider.setTracking(False)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 2, 2, 1, 1)

        self.retranslateUi(Cavity_View)
        self.buttonBox.accepted.connect(Cavity_View.accept)
        self.buttonBox.rejected.connect(Cavity_View.reject)
        QtCore.QMetaObject.connectSlotsByName(Cavity_View)

    def retranslateUi(self, Cavity_View):
        _translate = QtCore.QCoreApplication.translate
        Cavity_View.setWindowTitle(_translate("Cavity_View", "Cavity segmentation"))
        self.pushButton_2.setToolTip(_translate("Cavity_View", "Multiplication factor for the estimated variance at the seed point of the region growing. Defines acceptable pixel intensities for the region growing."))
        self.pushButton_2.setText(_translate("Cavity_View", "?"))
        self.label_4.setText(_translate("Cavity_View", "Size SE"))
        self.label_3.setText(_translate("Cavity_View", "Multiplier"))
        self.textEdit.setHtml(_translate("Cavity_View", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Cavity segmentation</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Segmentation of the cavity of wheat grain. The method is as follows:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1. Binarization (Otsu)</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2. Morphological opening with a structuring element of size \'Size SE\', in order to remove unwanted structures<br />3. Determination of the location of the cavity as the contour point with maximum distance to the convex hull of the binarized object</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">4. Computation of the distance transform (DT)</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">5. Region growing on the DT</p></body></html>"))
        self.pushButton.setToolTip(_translate("Cavity_View", "Size of the structuring element, in order to remove small unwanted structures."))
        self.pushButton.setText(_translate("Cavity_View", "?"))


