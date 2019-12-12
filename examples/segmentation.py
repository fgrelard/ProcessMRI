import nibabel as nib
import src.segmentation as segmentation
import src.maincontroller as mc
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.draw import circle
import src.imageio as io
from skimage.util import img_as_ubyte


def detect_circles(image, threshold=150, min_radius=10, max_radius=50):
    image_display = np.zeros_like(image)
    average = np.zeros(shape=(image.shape[0], 1))
    for i in range(image.shape[0]):
        center_x, center_y, radius = segmentation.detect_circle(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            circx, circy = circle(center_x, center_y, radius,shape=image_display[i, ...].shape)
            image_display[i, circx, circy] = image[i, circx, circy]
            average[i] = image_display[np.nonzero(image_display)].mean()
    return image_display


img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
img_data = img.get_fdata()
image = np.reshape(img_data, (img_data.shape[0], img_data.shape[1]) + (-1,), order='F')
image = image.transpose(2, 1, 0)

image8 = img_as_ubyte(image * 1.0 / image.max())
image8 = image8[6, ...]

cx, cy, r = segmentation.median_circle(image)
circx, circy = circle(cx, cy, r, shape=image8.shape)
image8[circy, circx] = 0

mask = segmentation.binarize(image8)
grain = segmentation.largest_connected_component(mask)
cond = np.where(grain == 0)

grain = image8.copy()
grain[cond] = 0
plt.imshow(grain)
plt.show()
image_display = segmentation.detect_cavity(grain)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_display, cmap=plt.cm.gray)
ax[1].imshow(grain)
plt.show()
