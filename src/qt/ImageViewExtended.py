import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox
import os
import pyqtgraph as pg


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
    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, *args):
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

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.label = pg.LabelItem(justify='right')
        self.scene.addItem(self.label)

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

    def export(self, filename):
        img = self.imageDisp[self.currentIndex, ...]
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
            self.export(root + ext)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No extension or wrong extension specified")
            msg.setInformativeText("Please specify a valid extension (.png, .svg or .pdf) ")
            msg.setWindowTitle("Wrong image format")
            msg.exec_()

    def levelsChanged(self):
        self.levelMin, self.levelMax = self.ui.histogram.getLevels()
