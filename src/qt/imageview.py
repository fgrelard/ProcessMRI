# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import nibabel as nib


# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')


image = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
image_data = image.get_fdata()
image_data = np.reshape(image_data, (image_data.shape[0], image_data.shape[1]) + (-1,))

## Display the data and assign each frame a time value from 1.0 to 3.0
imv.setImage(image_data, # xvals=np.linspace(1., 14., image_data.shape[2])
             axes= {'t':2, 'x':0, 'y':1}
)

## Set a custom color map
colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
# cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
# imv.setColorMap(cmap)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
