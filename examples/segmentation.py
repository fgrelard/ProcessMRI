import nibabel as nib
import src.segmentation as segmentation
import src.maincontroller as mc
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.draw import circle
from skimage.util import img_as_ubyte

img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
img_data = img.get_fdata()
image = np.reshape(img_data, (img_data.shape[0], img_data.shape[1]) + (-1,), order='F')

image = image.transpose()

def detect_circles(image, threshold=150, min_radius=10, max_radius=50):
    image_display = np.zeros_like(image)
    average = np.zeros(shape=(image.shape[0], 1))
    for i in range(image.shape[0]):
        center_x, center_y, radius = segmentation.detect_circle(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            circx, circy = circle(center_x, center_y, radius,shape=image_display[i, ...].shape)
            image_display[i, circy, circx] = image[i, circy, circx]
            average[i] = image_display[np.nonzero(image_display)].mean()
    return image_display

image8 = img_as_ubyte(image * 1.0 / image.max())
image8 = image8[8, ...]
#image_display = detect_circles(image8)
cx, cy, r = segmentation.detect_circle(image8, 150,10, 15)
circx, circy = circle(cx, cy, r, shape=image8.shape)
image8[circy, circx] = 0
mask = segmentation.detect_grain(image8)
grain=segmentation.largest_connected_component(mask)
cond = np.where(grain == 0)
grain = image8.copy()
grain[cond] = 0
image_display = segmentation.detect_cavity(grain)
plt.imshow(image_display, cmap=plt.cm.gray)
plt.show()