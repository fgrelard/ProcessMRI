import nibabel as nib
import src.segmentation as segmentation
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.draw import circle
import src.imageio as io
from skimage.util import img_as_ubyte
import sys

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
seg = segmentation.region_growing(image, (28,45,7))
sys.exit(0)

image = image.transpose(2, 1, 0)

coordinates = segmentation.closest_circle_to_median_circle(image, 10, 20)
# image = segmentation.remove_circle_3D(image, coordinates)
grain = segmentation.detect_grain_3D(image)
cavity = segmentation.detect_cavity_3D(grain, 3.5)

for i in range(cavity.shape[0]):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image[i, ...])
    ax[0].set_xlabel("Image")
    ax[1].imshow(grain[i, ...])
    ax[1].set_xlabel("Grain")
    ax[2].imshow(cavity[i, ...])
    ax[2].set_xlabel("Cavity")
    plt.show()
