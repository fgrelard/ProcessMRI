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



cx, cy, r = segmentation.median_circle(image)
image = segmentation.remove_circle(image, cx, cy, r+1)
grain = segmentation.detect_grain_3D(image)
cavity = segmentation.detect_cavity_3D(grain)
io.save_nifti(grain.transpose(2, 1, 0), "/mnt/d/IRM/nifti/BLE/250/50/50_grain.nii.gz")
for i in range(cavity.shape[0]):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cavity[i, ...])
    ax[1].imshow(image[i, ...])
    plt.show()
