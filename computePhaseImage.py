import sys
sys.path.append('./src/')
import argparse
import os
import nibabel as nib
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import cmath
import warnings
import skimage
from skimage.restoration import unwrap_phase
from compleximage import *
from temporalphasecorrection import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 4D MRI image (x,y,z,echoes), extension: NIFTI)")
parser.add_argument("-o", "--output", help="Output Temporally Phase Corrected Magnitude image (x,y,z), extension: NIFTI")
args = parser.parse_args()

warnings.simplefilter('ignore', np.RankWarning)

input = args.input
output = args.output

filename = os.path.splitext(input)[0]
img  = nib.load(input)
img_data = img.get_fdata()

path = os.path.dirname(input)
with open(os.path.join(path,'1.json')) as f:
    data = json.load(f)


echotime = [item for sublist in data['EchoTime'] for item in sublist]
complex_corrected_image = correct_phase_temporally(echotime, img_data, 2)
#draw_phase_repartition(to_complex(img_data), complex_corrected_image)

out_img_data = complex_to_magnitude(to_complex(img_data))
out_img_data = complex_corrected_image.real
out_img = nib.Nifti1Image(out_img_data, np.eye(4))
out_img.to_filename(output)
