import unittest
import nibabel as nib
import importlib.machinery
import src.segmentation as segmentation
import src.maincontroller as mc
import numpy as np
import matplotlib.pyplot as plt


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        img = nib.load("/mnt/d/IRM/nifti/BLE/250/50/50_subscan_1.nii.gz")
        img_data = img.get_fdata()
        img2 = np.reshape(img_data, (img_data.shape[0], img_data.shape[1]) + (-1,), order='F')

        img2 = img2.transpose()
        print(img2.shape)
        self.image = img2

    def test_segmentation_tube(self):
        x, y, r = segmentation.detect_tube(self.image)
        plt.imshow(self.image[8,...])
        plt.show()
        print(x,y,r)

if __name__ == "__main__":
    unittest.main()
