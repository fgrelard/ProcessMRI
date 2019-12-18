# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/mainview.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFrame
import src.ImageViewExtended as ive


class Ui_MainView(object):
    def setupUi(self, MainView):
        MainView.setObjectName("MainView")
        MainView.setEnabled(True)
        MainView.resize(810, 593)
        self.centralwidget = QtWidgets.QWidget(MainView)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)

        self.imageview = ive.ImageViewExtended(parent=self.centralwidget)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setText("Running...")
        self.progressBar = QtWidgets.QProgressBar(self.gridLayoutWidget)
        self.stopButton = QtWidgets.QPushButton(self.gridLayoutWidget)

        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setFrameStyle(QFrame.NoFrame)
        self.labelCombo = QtWidgets.QLabel(self.gridLayoutWidget)
        self.horizontalSpace = QtWidgets.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        self.combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.menubar = QtWidgets.QMenuBar(MainView)
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.actionExit = QtWidgets.QAction(MainView)

        self.actionSave = QtWidgets.QAction(MainView)

        self.statusbar = QtWidgets.QStatusBar(MainView)

        self.menuHelp = QtWidgets.QMenu(self.menubar)

        self.menuProcess = QtWidgets.QMenu(self.menubar)

        self.menuAnalyze = QtWidgets.QMenu(self.menubar)

        self.menuOpen = QtWidgets.QMenu(self.menuFile)

        self.actionBruker_directory = QtWidgets.QAction(MainView)
        self.actionDenoising_NL_means = QtWidgets.QAction(MainView)

        self.actionExponential_fitting = QtWidgets.QAction(MainView)

        self.actionUser_manual_FR = QtWidgets.QAction(MainView)

        self.actionDenoising_TPC = QtWidgets.QAction(MainView)

        self.actionNifti = QtWidgets.QAction(MainView)

        self.actionHoughTransform = QtWidgets.QAction(MainView)

        self.actionSegmentCavity = QtWidgets.QAction(MainView)

        self.actionSegmentGrain = QtWidgets.QAction(MainView)

        self.menuFile = QtWidgets.QMenu(self.menubar)

        self.configure(MainView)

        self.retranslateUi(MainView)
        QtCore.QMetaObject.connectSlotsByName(MainView)

    def show_run(self):
        self.label.show()
        self.progressBar.show()
        self.stopButton.show()

    def hide_run(self):
        self.label.hide()
        self.progressBar.hide()
        self.stopButton.hide()

    def configure(self, MainView):
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 750, 500))

        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout()

        self.gridLayout.setContentsMargins(20, 10, 0, 0)
        self.gridLayout.setObjectName("gridLayout")


        self.progressBar.setEnabled(True)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.progressBar.setVisible(False)
        self.progressBar.setTextVisible(True)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.progressBar, 4, 0, 1, 1)

        self.stopButton.setText("Stop")
        self.stopButton.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.gridLayout.addWidget(self.stopButton, 5, 0, 1, 1)

        self.textEdit.setAcceptDrops(False)
        self.textEdit.setAutoFillBackground(True)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 1)

        self.labelCombo.setText("Current image: ")
        self.combobox.setFixedWidth(100)
        self.combobox.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self.combobox.addItem("No image")
        self.hLayout = QtWidgets.QHBoxLayout()
        self.hLayout.addWidget(self.labelCombo)
        self.hLayout.addWidget(self.combobox)
        self.hLayout.addStretch()
        self.hLayout.setAlignment(QtCore.Qt.AlignLeft)

        self.gridLayout.addLayout(self.hLayout, 0, 1, 1, 1)

        # self.imageview.ui.menuBtn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # self.imageview.ui.roiBtn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # self.imageview.ui.histogram.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.imageview.ui.graphicsView.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.imageview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.imageview.ui.histogram.vb.setFixedWidth(2)
        # self.imageview.ui.histogram.vb.setMinimumWidth(2)
        self.gridLayout.addWidget(self.imageview, 1, 1, 1, 1)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        MainView.setCentralWidget(self.centralwidget)



        self.menubar.setGeometry(QtCore.QRect(0, 0, 810, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile.setObjectName("menuFile")
        self.menuOpen.setObjectName("menuOpen")
        self.menuProcess.setObjectName("menuProcess")
        self.menuAnalyze.setObjectName("menuAnalyze")
        self.menuHelp.setObjectName("menuHelp")
        MainView.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")
        MainView.setStatusBar(self.statusbar)
        self.actionSave.setObjectName("actionSave")
        self.actionExit.setObjectName("actionExit")
        self.actionBruker_directory.setObjectName("actionBruker_directory")
        self.actionNifti.setObjectName("actionNifti")
        self.actionExponential_fitting.setObjectName("actionExponential_fitting")
        self.actionDenoising_TPC.setObjectName("actionDenoising_TPC")
        self.actionDenoising_NL_means.setObjectName("actionDenoising_NL_means")
        self.actionUser_manual_FR.setObjectName("actionUser_manual_FR")
        self.menuOpen.addAction(self.actionBruker_directory)
        self.menuOpen.addAction(self.actionNifti)
        self.menuFile.addAction(self.menuOpen.menuAction())
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuProcess.addAction(self.actionExponential_fitting)
        self.menuProcess.addAction(self.actionDenoising_TPC)
        self.menuProcess.addAction(self.actionDenoising_NL_means)
        self.menuAnalyze.addAction(self.actionHoughTransform)
        self.menuAnalyze.addAction(self.actionSegmentGrain)
        self.menuAnalyze.addAction(self.actionSegmentCavity)
        self.menuHelp.addAction(self.actionUser_manual_FR)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuProcess.menuAction())
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())


    def retranslateUi(self, MainView):
        _translate = QtCore.QCoreApplication.translate
        MainView.setWindowTitle(_translate("MainView", "ProcessMRI"))
        self.textEdit.setHtml(_translate("MainView", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">ProcessMRI</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Simple tools to process MRI images.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">First open an image with &quot;File/Open&quot;.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Second, use processing tools in &quot;Process&quot;:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- multi-exponential fit</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- denoising with temporal phase correction</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- denoising with non-local means</p></body></html>"))
        self.textEdit.setStyleSheet("background: rgba(0,0,0,0%)")
        self.textEdit.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.menuFile.setTitle(_translate("MainView", "File"))
        self.menuOpen.setTitle(_translate("MainView", "Open"))
        self.menuProcess.setTitle(_translate("MainView", "Process"))
        self.menuAnalyze.setTitle(_translate("MainView", "Analyze"))
        self.menuHelp.setTitle(_translate("MainView", "Help"))
        self.actionSave.setText(_translate("MainView", "Save Nifti"))
        self.actionExit.setText(_translate("MainView", "Exit"))
        self.actionBruker_directory.setText(_translate("MainView", "Bruker directory"))
        self.actionNifti.setText(_translate("MainView", "Nifti"))
        self.actionExponential_fitting.setText(_translate("MainView", "Exponential fitting"))
        self.actionDenoising_TPC.setText(_translate("MainView", "Denoising TPC"))
        self.actionDenoising_NL_means.setText(_translate("MainView", "Denoising NL-means"))
        self.actionHoughTransform.setText(_translate("MainView", "Hough transform"))
        self.actionSegmentGrain.setText(_translate("MainView", "Largest component"))
        self.actionSegmentCavity.setText(_translate("MainView", "Segment cavity"))
        self.actionUser_manual_FR.setText(_translate("MainView", "User manual (FR)"))
