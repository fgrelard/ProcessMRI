import numpy as np
import src.measurements as measurements
import nibabel as nib





img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
img_data = img.get_fdata()
image = np.reshape(img_data, (img_data.shape[0], img_data.shape[1]) + (-1,), order='F')
# image = image.transpose(2, 1, 0)

print("Area pixels: ", measurements.area_pixels(image))
print("Area unit: ", measurements.area_unit(image))
print("Area unit (112): ", measurements.area_unit(image, (0.5,0.5, 0.05)))
print("Average: ", measurements.average_value(image))
print("Min: ", measurements.min_value(image))
print("Max: ", measurements.max_value(image))
