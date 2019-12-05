import pyqtgraph as pg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QApplication
import os
import pyqtgraph as pg
import time
class WorkerExport(QtCore.QObject):
    signal_end = QtCore.pyqtSignal()

    def __init__(self, ive, path):
        super().__init__()
        self.ive = ive
        self.path = path
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        print(self.ive.imageDisp.shape[0])
        indexExportSlice = 0
        while indexExportSlice < self.ive.imageDisp.shape[0]:
            QApplication.processEvents()
            if self.is_abort:
                break
            self.ive.export(self.path + os.path.sep + str(self.ive.currentIndex) + ".png", indexExportSlice)
            indexExportSlice += 1
        self.signal_end.emit()

    def abort(self):
        self.is_abort = True



def addNewGradientFromMatplotlib( name):
    gradient = cm.get_cmap(name)
    L = []
    nb = 10
    for i in range(nb):
        normI = float(i/(nb-1))
        elemColor = ((normI, tuple(int(elem*255) for elem in gradient(normI))))
        L.append(elemColor)
    pg.graphicsItems.GradientEditorItem.Gradients[name] = {'ticks':L, 'mode': 'rgb'}

class ImageViewExtended(pg.ImageView):

    signal_abort = QtCore.pyqtSignal()

    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, *args):
        pg.setConfigOptions(imageAxisOrder='row-major')
        addNewGradientFromMatplotlib("jet")
        addNewGradientFromMatplotlib("viridis")
        addNewGradientFromMatplotlib("plasma")
        addNewGradientFromMatplotlib("inferno")
        addNewGradientFromMatplotlib("magma")
        addNewGradientFromMatplotlib("cividis")
        super().__init__(parent, name, view, imageItem, *args)
        self.timeLine.setPen('g')

        self.ui.histogram.sigLevelsChanged.connect(self.levelsChanged)
        self.ui.histogram.gradient.loadPreset("viridis")
        self.ui.histogram.gradient.updateGradient()
        self.ui.histogram.gradientChanged()
        self.hide_partial()

        self.label = pg.LabelItem(justify='right')
        self.scene.addItem(self.label)
        self.scene.sigMouseMoved.connect(self.on_hover_image)
        
        self.threads = []

        self.mouse_x = 0
        self.mouse_y = 0


    def hide_partial(self):
        self.ui.roiBtn.hide()
        self.ui.label_4.hide()
        self.ui.label_8.hide()
        self.ui.label_9.hide()
        self.ui.label_10.hide()
        self.ui.normXBlurSpin.hide()
        self.ui.normYBlurSpin.hide()
        self.ui.normTBlurSpin.hide()
        self.ui.normFrameCheck.hide()
        self.ui.gridLayout_2.addWidget(self.ui.normTimeRangeCheck, 1, 2, 1, 1)

    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True):
        super().setImage(img, autoRange, autoLevels, levels, axes, xvals, pos, scale, transform, autoHistogramRange)
        self.ui.roiPlot.setMouseEnabled(True, True)
        max_t = img.shape[0]
        self.normRgn.setRegion((1, max_t//2))

    def on_hover_image(self, evt):
        pos = evt
        mousePoint = self.view.mapSceneToView(pos)
        self.mouse_x = int(mousePoint.x())
        self.mouse_y = int(mousePoint.y())
        image = self.imageDisp
        if image is None:
            return
        self.update_label()

    def update_label(self):
        if not (self.mouse_x >= 0 and self.mouse_x < self.imageDisp.shape[-1] and
            self.mouse_y >= 0 and self.mouse_y < self.imageDisp.shape[-2]):
            self.mouse_x = 0
            self.mouse_y = 0
        self.display_label()

    def display_label(self):
        position = "(" + str(self.mouse_x) + ", " + str(self.mouse_y)
        if self.imageDisp.ndim == 2:
            value = self.imageDisp[(self.mouse_y, self.mouse_x)]
        if self.imageDisp.ndim == 3:
            position += ", " + str(self.currentIndex) + ")"
            value = str(self.imageDisp[(self.currentIndex,self.mouse_y,self.mouse_x)])
        self.label.setText("<span>" + position + "</span><span style='font-weight:bold; color: green;'>: " + value + "</span>")

    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.update_label()

    def timeLineChanged(self):
        super().timeLineChanged()
        self.update_label()

    def getProcessedImage(self):
        if self.imageDisp is None:
            image = self.normalize(self.image)
            self.imageDisp = image
            if self.axes['t'] is not None:
                curr_img = self.imageDisp[self.currentIndex, ...]
                self.levelMin, self.levelMax = np.amin(curr_img), np.amax(curr_img)
            else:
                self.levelMin, self.levelMax = list(map(float, self.quickMinMax(self.imageDisp)))
        return self.imageDisp


    def normalize(self, image):
        """
        Process *image* using the normalization options configured in the
        control panel.

        This can be repurposed to process any data through the same filter.
        """
        if self.ui.normOffRadio.isChecked():
            return image

        div = self.ui.normDivideRadio.isChecked()
        norm = image.view(np.ndarray).copy()
        #if div:
            #norm = ones(image.shape)
        #else:
            #norm = zeros(image.shape)
        if div:
            norm = norm.astype(np.float32)

        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.normRgn.lines[0])
            (eind, end) = self.timeIndex(self.normRgn.lines[1])
            #print start, end, sind, eind
            n = image[sind:eind+1].mean(axis=0)
            n.shape = (1,) + n.shape
            if div:
                norm /= n
            else:
                norm -= n

        if self.ui.normFrameCheck.isChecked() and image.ndim == 3:
            n = image.mean(axis=1).mean(axis=1)
            n.shape = n.shape + (1, 1)
            if div:
                norm /= n
            else:
                norm -= n

        if self.ui.normROICheck.isChecked() and image.ndim == 3:
            if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
                norm = image.view(np.ndarray).copy()
                if div:
                    norm = norm.astype(np.float32)
                roi = norm[sind:eind+1]
                n = self.normRoi.getArrayRegion(roi, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1).mean(axis=0)
            else:
                n = self.normRoi.getArrayRegion(norm, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1)
                n = n[:,np.newaxis,np.newaxis]

            #print start, end, sind, eind
            if div:
                norm /= n
            else:
                norm -= n

        return norm

    def export(self, filename, index):
        if self.imageDisp.ndim == 2:
            img = self.imageDisp
        else:
            img = self.imageDisp[index, ...]
        current_cm = self.ui.histogram.gradient.colorMap().getColors()
        current_cm = current_cm.astype(float)
        current_cm /= 255.0
        nb = len(current_cm[...,0])
        red, green, blue = [], [], []
        for i in range(nb):
            red.append([float(i/(nb-1)), current_cm[i, 0], current_cm[i, 0]])
            green.append([float(i/(nb-1)), current_cm[i, 1], current_cm[i, 1]])
            blue.append([float(i/(nb-1)), current_cm[i, 2], current_cm[i, 2]])
        cdict = {'red': np.array(red),
                 'green': np.array(green),
                 'blue': np.array(blue)}
        newcmp = colors.LinearSegmentedColormap("current_cmap", segmentdata=cdict)
        pos = plt.imshow(img, cmap=newcmp)
        plt.colorbar()
        plt.clim(self.levelMin, self.levelMax)
        plt.axis('off')
        # plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()
        plt.close()


    def exportClicked(self):
        fileName, image_format = QtGui.QFileDialog.getSaveFileName(None, "Save image as...", "", "PNG images (.png);;Portable Document Format (.pdf);; Scalable Vector Graphics (.svg)")
        if not fileName:
            return
        root, ext = os.path.splitext(fileName)
        if not ext:
            if "png" in image_format:
                ext = ".png"
            if "svg" in image_format:
                ext = ".svg"
            if "pdf" in image_format:
                ext = ".pdf"
        if ext == ".png" or ext == ".svg" or ext == ".pdf":
            self.export(root + ext, self.currentIndex)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No extension or wrong extension specified")
            msg.setInformativeText("Please specify a valid extension (.png, .svg or .pdf) ")
            msg.setWindowTitle("Wrong image format")
            msg.exec_()

    def exportSlicesClicked(self):
        path = QtGui.QFileDialog.getExistingDirectory(None, "Select a directory", "")
        if len(self.threads) > 0:
            self.signal_abort.emit()
            for thread, worker in self.threads:
                thread.quit()
                thread.wait()
            print("quit")

        self.previousIndex = self.currentIndex
        worker = WorkerExport(self, path)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        worker.signal_end.connect(self.reset_index)
        self.signal_abort.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))

    def reset_index(self):
        self.currentIndex = self.previousIndex


    def levelsChanged(self):
        self.levelMin, self.levelMax = self.ui.histogram.getLevels()

    def buildMenu(self):
        super().buildMenu()
        self.exportSlicesAction = QtGui.QAction("Export all slices", self.menu)
        self.exportSlicesAction.triggered.connect(self.exportSlicesClicked)
        self.menu.addAction(self.exportSlicesAction)
