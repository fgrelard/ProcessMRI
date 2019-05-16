from __future__ import absolute_import

import sys
sys.path.append('../src/')

import unittest
import nibabel as nib
import src.bruker2nifti as b2n
import numpy as np
import numpy.testing as nptest
import json

class TestBruker2Nifti(unittest.TestCase):
    def test_convert(self):
        b2n.convert("/mnt/d/IRM/raw/50")

if __name__ == "__main__":
    unittest.main()
