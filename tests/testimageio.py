from __future__ import absolute_import

import sys
sys.path.append('../src/')

import unittest
import nibabel as nib
import src.imageio as io
import numpy as np
import numpy.testing as nptest
import json

class TestImageIO(unittest.TestCase):
    def test_open_generic_image(self):
        l = io.open_generic_image("/mnt/d/IRM/raw/50")

    def test_bruker2nifti(self):
        io.bruker2nifti("/mnt/d/IRM/raw/50")

    def test_open_metadata(self):
        io.open_metadata("/mnt/d/IRM/raw/50/nifti/50_subscan_1.nii.gz")

    def test_open_bruker(self):
        l = io.open_bruker("/mnt/d/IRM/raw/50")

        if __name__ == "__main__":
    unittest.main()
