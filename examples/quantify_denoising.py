import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration as skrestore
import nibabel as nib
import scipy.ndimage.filters as filters

def stddev(val_list):
    return np.mean(val_list)

def std_intensity(image, w):
    std_image = filters.generic_filter(image, function=stddev,
                                       size=3, mode='constant', cval=0)
    return std_image


def sigma_mean_ratio(image):
    mean = np.mean(image)
    sigma = np.mean(skrestore.estimate_sigma(image, multichannel=True))
    return sigma/mean

img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/processed/denoised_test.nii")
img_data = img.get_fdata()
#std_image = std_intensity(img_data, 3)
sigma_est = sigma_mean_ratio(img_data)
sigma_est_0 = sigma_mean_ratio(img_data[:,:,:,0])
sigma_est_7 = sigma_mean_ratio(img_data[:,:,:,7])

print("Sigma noise all=", sigma_est, ", first echo=", sigma_est_0, ", last echo=", sigma_est_7)
