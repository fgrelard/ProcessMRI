import sys
sys.path.append('../src/')

import unittest
import nibabel as nib
import src.exponentialfit as ef
import numpy as np
import numpy.testing as nptest
import json

class TestExponentialFit(unittest.TestCase):
    def setUp(self):
        with open("/mnt/d/IRM/nifti/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/1.json") as f:
            data = json.load(f)
        self.echotime = [item for sublist in data['EchoTime'] for item in sublist]

    def test_denoise_image(self):
        img = nib.load("/mnt/d/IRM/nifti/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/1.nii")
        img_data = img.get_fdata()
        denoised = ef.denoise_image(img_data[:,:,5,:])
        out_img = nib.Nifti1Image(denoised, np.eye(4))
        out_img.to_filename("/mnt/d/IRM/nifti/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/denoise_test.nii")

    def test_estimation_2D_density_image(self):
        img = nib.load("/mnt/d/IRM/nifti/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/1.nii")
        img_data = img.get_fdata()
        out_img_data, t2 = ef.exponentialfit_image(self.echotime, img_data[:,:,5,:], 2000)
        out_img = nib.Nifti1Image(out_img_data, np.eye(4))
        out_img.to_filename("/mnt/d/IRM/nifti/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/correction_test.nii")

if __name__ == "__main__":
    unittest.main()
