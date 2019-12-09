import pyqtgraph as pg
import numpy as np
import matplotlib

#Allows to use QThreads without freezing
#the main application
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
    """
    Worker for the export of all the slices
    corresponding to the image
    """
    signal_start = QtCore.pyqtSignal()
    signal_end = QtCore.pyqtSignal()
    signal_progress = QtCore.pyqtSignal(int)

    def __init__(self, ive, path):
        """
        Parameters
        ----------
        ive: ImageViewExtended
            the image view
        path: str
            path to the filename
        """
        super().__init__()
        self.ive = ive
        self.path = path
        self.is_abort = False

    @QtCore.pyqtSlot()
    def work(self):
        """
        Export function with progress value signalled
        """
        indexExportSlice = 0
        length = self.ive.imageDisp.shape[0]
        self.signal_start.emit()
        while indexExportSlice < length:
            QApplication.processEvents()
            if self.is_abort:
                break
            self.ive.export(self.path + os.path.sep + str(indexExportSlice) + ".png", indexExportSlice)
            indexExportSlice += 1
            progress = float(indexExportSlice / length * 100)
            self.signal_progress.emit(progress)
        self.signal_end.emit()

    def abort(self):
        self.is_abort = True



def addNewGradientFromMatplotlib( name):
    """
    Generic function to add a gradient from a
    matplotlib colormap

    Parameters
    ----------
    name: str
        name of matplotlib colormap
    """
    gradient = cm.get_cmap(name)
    L = []
    nb = 10
    for i in range(nb):
        normI = float(i/(nb-1))
        elemColor = ((normI, tuple(int(elem*255) for elem in gradient(normI))))
        L.append(elemColor)
    pg.graphicsItems.GradientEditorItem.Gradients[name] = {'ticks':L, 'mode': 'rgb'}

class ImageViewExtended(pg.ImageView):
    """
    Image view extending pyqtgraph.ImageView

    Attributes
    ----------
    label: pg.LabelItem
        Display pixel values and positions
    threads: list
        List of threads
    mouse_x: int
        Mouse position x
    mouse_y: int
        Mouse position y
    """

    signal_abort = QtCore.pyqtSignal()
    signal_progress_export = QtCore.pyqtSignal(int)
    signal_start_export = QtCore.pyqtSignal()
    signal_end_export = QtCore.pyqtSignal()

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
        """
        Hide some elements from the parent GUI
        """
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
        """
        Sets a new image

        When changing an image, tries to keep the old z-index

        Changes the wheel-event from zoom-out
        to slice change
        """

        #Saves previous z-index
        previousIndex = self.currentIndex
        is_shown = False
        if self.imageDisp is not None:
            previousShape = self.imageDisp.shape
            is_shown = True

        super().setImage(img, autoRange, autoLevels, levels, axes, xvals, pos, scale, transform, autoHistogramRange)

        #Changes wheel event
        self.ui.roiPlot.setMouseEnabled(True, True)
        self.ui.roiPlot.wheelEvent = self.roi_scroll_bar
        max_t = img.shape[0]
        self.normRgn.setRegion((1, max_t//2))
        if not is_shown:
            return
        #Shows image at previous z-index if in range
        if previousIndex < self.imageDisp.shape[0]:
            self.setCurrentIndex(previousIndex)

    def roi_scroll_bar(self, ev):
        """
        Changes the z-index of the 3D image
        when scrolling the z-bar

        Parameters
        ----------
        ev: QWheelEvent
            the wheel event
        """
        new_index = self.currentIndex + 1 if ev.angleDelta().y() < 0 else self.currentIndex - 1
        self.setCurrentIndex(new_index)


    def on_hover_image(self, evt):
        """
        Updates the mouse positions and pixel values
        when hovering over the image

        Parameters
        ----------
        evt: QMouseEvent
            the mouse event
        """
        pos = evt
        mousePoint = self.view.mapSceneToView(pos)
        self.mouse_x = int(mousePoint.x())
        self.mouse_y = int(mousePoint.y())
        image = self.imageDisp
        if image is None:
            return
        self.update_label()

    def update_label(self):
        """
        Updates the label with mouse position
        and pixel values relative to the image
        """
        if not (self.mouse_x >= 0 and self.mouse_x < self.imageDisp.shape[-1] and
            self.mouse_y >= 0 and self.mouse_y < self.imageDisp.shape[-2]):
            self.mouse_x = 0
            self.mouse_y = 0
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

        Makes it so ROI + timeline normalization is the
        intersection of both as opposed to timeline being
        favored in the parent class
        """
        if self.ui.normOffRadio.isChecked():
            return image

        div = self.ui.normDivideRadio.isChecked()
        norm = image.view(np.ndarray).copy()
        if div:
            norm = norm.astype(np.float32)

        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.normRgn.lines[0])
            (eind, end) = self.timeIndex(self.normRgn.lines[1])
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
            #If ROI checked, only work on this part of the image
            if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
                norm = image.view(np.ndarray).copy()
                if div:
                    norm = norm.astype(np.float32)
                roi = norm[sind:eind+1]
                n = self.normRoi.getArrayRegion(roi, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1).mean(axis=0)
            #Other case : no ROI checked
            else:
                n = self.normRoi.getArrayRegion(norm, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1)
                n = n[:,np.newaxis,np.newaxis]

            if div:
                norm /= n
            else:
                norm -= n

        return norm

    def export(self, filename, index):
        """
        Export image view to file through Matplotlib
        Saves a scalebar on the side
        Accepted formats are .pdf, .png and .svg

        Parameters
        ----------
        self: type
            description
        filename: str
            image name
        index: int
            z-index of the image to save
        """
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
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()
        plt.close()


    def exportClicked(self):
        """
        Called when the "Export" button is clicked
        """
        fileName, image_format = QtGui.QFileDialog.getSaveFileName(self.parentWidget(), "Save image as...", "", "PNG images (.png);;Portable Document Format (.pdf);; Scalable Vector Graphics (.svg)")
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
        """
        Called when the "Export all slices" button is clicked
        """
        path = QtGui.QFileDialog.getExistingDirectory(self.parentWidget(), "Select a directory", "")
        if len(self.threads) > 0:
            self.signal_abort.emit()
            for thread, worker in self.threads:
                thread.quit()
                thread.wait()
        if not path:
            return
        self.previousIndex = self.currentIndex
        worker = WorkerExport(self, path)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        worker.signal_end.connect(self.reset_index)
        worker.signal_start.connect(self.signal_start_export.emit)
        worker.signal_end.connect(self.signal_end_export.emit)
        worker.signal_progress.connect(lambda progress: self.signal_progress_export.emit(progress))
        self.signal_abort.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))

    def reset_index(self):
        """
        Called when the end signal of export slices is emitted
        """
        self.currentIndex = self.previousIndex

    def levelsChanged(self):
        """
        Called when the levels of the histogram are changed
        """
        self.levelMin, self.levelMax = self.ui.histogram.getLevels()

    def buildMenu(self):
        """
        Adds the "Export all slices" option to the menu
        """
        super().buildMenu()
        self.exportSlicesAction = QtGui.QAction("Export all slices", self.menu)
        self.exportSlicesAction.triggered.connect(self.exportSlicesClicked)
        self.menu.addAction(self.exportSlicesAction)
