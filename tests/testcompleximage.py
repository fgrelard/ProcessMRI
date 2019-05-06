import unittest
import nibabel as nib
import importlib.machinery

spec = importlib.util.spec_from_file_location("module.name", "/home/fgrelard/src/readParametricMRI/compleximage.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)

class TestComplexImage(unittest.TestCase):
    def setUp(self):
        img = nib.load("/mnt/d/IRM/nifti/7/BLE RECITAL/1_BLE 250DJ/50_7_MGE/1.nii")
        img_data = img.get_fdata()
        self.image = img_data

    def test_complex_to_phase(self):
        img = ci.complex_to_phase(self.image)

    def test_to_complex(self):
        img = ci.to_complex(self.image)
        print(img)

if __name__ == "__main__":
    unittest.main()
