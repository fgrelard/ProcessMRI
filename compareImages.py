import argparse
import os
import nibabel as nib
import numpy as np
import math
import json
import matplotlib.pyplot as plt

def compare_at_point(first, second, x, y):
    value_first = first.get_data()[y]
    value_second = second.get_data()[y]
    print(np.std(value_first), np.std(value_second))
    plt.plot(x, value_first, 'o', x, value_second, 'x', fillstyle="full")
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--first", help="4D MRI image, extension: NIFTI)")
parser.add_argument("-s", "--second", help="4D MRI image, extension: NIFTI")
args = parser.parse_args()

first = args.first
second = args.second

firstImage  = nib.load(first)
firstImageData = firstImage.get_fdata()

secondImage  = nib.load(second)
secondImageData = secondImage.get_fdata()

path = os.path.dirname(first)
with open(os.path.join(path,'1.json')) as f:
        data = json.load(f)

echoTime = [item for sublist in data['EchoTime'] for item in sublist]
compare_at_point(firstImage, secondImage, echoTime[:8], (15,20,3))
