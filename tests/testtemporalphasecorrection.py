import sys
sys.path.append('../src/')

import unittest
import nibabel as nib
import importlib.machinery
import compleximage as ci
import temporalphasecorrection as tpc
import numpy as np
import numpy.testing as nptest

class TestTemporalPhaseCorrection(unittest.TestCase):
    def setUp(self):
        img = nib.load("/mnt/d/IRM/nifti/7/BLE RECITAL/1_BLE 250DJ/50_7_MGE/1.nii")
        img_data = img.get_fdata()
        self.image = img_data

    def test_correct_phase_1d(self):
        dn = tpc.correct_phase_1d([1.2634,3.2634,5.2634,7.2634],
                                  [complex(-4302.3, 4106.4),
                                   complex(-2776.3, -4215.5),
                                   complex(3478.0, 2246.9),
                                   complex(1139.4, 3015.4)],
                                  3,
                                  1)
        nptest.assert_array_almost_equal(dn, [complex(5947.46216886,3.63797881e-11),
                                              complex(5047.60160274,2.41016096e-11),
                                              complex(4140.65738863,-1.61435310e-11j),
                                              complex(3223.48716765,9.00399755e-11j)])



    def test_draw_phase_repartition(self):
        complex_img = ci.to_complex(self.image[:,:,4,:])
        img = tpc.correct_phase_temporally(range(1, 9), self.image[:,:,4,:], 3)
        tpc.draw_phase_repartition(complex_img, img)

if __name__ == "__main__":
    unittest.main()
