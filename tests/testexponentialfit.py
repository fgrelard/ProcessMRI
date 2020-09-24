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
        with open("/mnt/d/IRM/nifti/BLE/250/50/old/1/BLE RECITAL/1_BLE 250DJ/50_1_MGE/1.json") as f:
            data = json.load(f)
        self.echotime = [item for sublist in data['EchoTime'] for item in sublist]
        self.good = (43,68,9)
        self.bad = (16, 55, 7)

    def test_denoise_image(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
        img_data = img.get_fdata()
        denoised = ef.denoise_image(img_data, 5, 6, 1.5)
        out_img = nib.Nifti1Image(denoised, np.eye(4))
        out_img.to_filename("/mnt/d/IRM/nifti/BLE/250/50/processed/denoised_test.nii")

    def test_estimation_2D_density_image(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
        img_data = img.get_fdata()
        out_img_data, t2 = ef.exponentialfit_image(self.echotime, img_data[:,:,5,:], 2000)
        out_img = nib.Nifti1Image(out_img_data, np.eye(4))
        out_img.to_filename("/mnt/d/IRM/nifti/BLE/250/50/processed/correction_test.nii")

    def test_fit_exponential_linear_regression(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/processed/50_subscan_1_denoised.nii")
        img_data = img.get_fdata()
        pixel_values = img_data[self.good + (slice(None),)]
        fit, _ = ef.fit_exponential_linear_regression(self.echotime, pixel_values)
        f = ef.n_exponential_function(np.array(self.echotime), *fit)
        print(f)
        ef.plot_values(self.echotime,pixel_values,1, fit, 0, ef.n_exponential_function)

    def test_fit_exponential_piecewise_linear_regression(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/processed/50_subscan_1_denoised.nii")
        img_data = img.get_fdata()
        pixel_values = img_data[self.good + (slice(None),)]
        fit, _ = ef.fit_exponential_piecewise_linear_regression(self.echotime, pixel_values)
        f = ef.n_exponential_function(np.array(self.echotime), *fit)
        print(f)
        ef.plot_values(self.echotime,pixel_values,1, fit, 0, ef.n_exponential_function)

    def test_fit_exponential_piecewise_linear_regression_2(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/processed/50_subscan_1_denoised.nii")
        img_data = img.get_fdata()
        pixel_values = img_data[self.good + (slice(None),)]
        fit, _ = ef.fit_exponential_piecewise_linear_regression_2(self.echotime, pixel_values)
        f = ef.n_exponential_function(np.array(self.echotime), *fit)
        print(f)
        ef.plot_values(self.echotime,pixel_values,1, fit, 0, ef.n_exponential_function)

    def test_fit_exponential(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/processed/50_subscan_1_denoised.nii")
        img_data = img.get_fdata()
        pixel_values = img_data[self.bad + (slice(None),)]
        n=1
        p0 = ef.n_to_p0(n, pixel_values[0])
        fit, _ = ef.fit_exponential(self.echotime, pixel_values, p0)
        f = ef.n_exponential_function(n, *fit)
        ef.plot_values(self.echotime,pixel_values,1, fit, 0, ef.n_exponential_function)



if __name__ == "__main__":
    unittest.main()
