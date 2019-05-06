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

# def separate_real_imaginary(img_data):
#     real = []
#     imaginary = []
#     ri = img_data.shape[3]
#     for l in range(ri):
#         image = img_data[:, :, :, l]
#         if l < 8:
#             real.append(image)
#         else:
#             imaginary.append(image)
#     return real, imaginary


def unwrap_phases(phase_image):
    x = phase_image.shape[0]
    y = phase_image.shape[1]
    z = phase_image.shape[2]
    w = phase_image.shape[3]
    unwrapped_image = np.zeros(shape=(x, y, z, w))
    for k in range(z):
        for l in range(w):
            unwrapped_image_2D = unwrap_phase(skimage.img_as_float(phase_image[:,:,k,l]))
            unwrapped_image[:,:,k,l] = unwrapped_image_2D
    return unwrapped_image


def phase_correct(echotimes, decays, order, noise):
    phases = [cmath.phase(decay) for decay in decays]
    phases_unwrap = np.unwrap(phases)
    weighting_vector = np.ones(shape=(len(phases_unwrap)))
    p = np.polyfit(echotimes, phases_unwrap, order, w=weighting_vector)
    f = np.polyval(p, echotimes)
    decay_new = np.multiply(decays, np.exp(-1j*np.transpose(f)))
    return decay_new

# def to_complex(img_data):
#     x = img_data.shape[0]
#     y = img_data.shape[1]
#     z = img_data.shape[2]
#     w = img_data.shape[3]/2
#     complex_img = np.zeros(shape=(x, y, z, w), dtype=complex)
#     real, imaginary = separate_real_imaginary(img_data)
#     for i in range(x):
#         for j in range(y):
#             for k in range(z):
#                 for l in range(w):
#                     complex_img[i, j, k, l] = complex(real[l][i, j, k], imaginary[l][i, j, k])
#     return complex_img

def correct_phase_temporally(echotimes, img_data, order):
    x = img_data.shape[0]
    y = img_data.shape[1]
    z = img_data.shape[2]
    w = img_data.shape[3]/2

    out_img_data = np.zeros(shape=(x, y, z, w), dtype=complex)
    complex_img_data = to_complex(img_data)
    even_echotime = echotimes[:8:2]
    odd_echotime = echotimes[1:8:2]
    even_complex_img = complex_img_data[:, :, :, ::2]
    odd_complex_img = complex_img_data[:, :, :, 1::2]
    phase_image = complex_to_phase(complex_img_data)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                tpc_even = phase_correct(even_echotime, even_complex_img[i, j, k], order, 1)
                tpc_odd = phase_correct(odd_echotime, odd_complex_img[i, j, k], order, 1)
                for l in range(w):
                    if l % 2 == 0:
                        out_img_data[i, j, k, l] = tpc_even[l/2]
                    else:
                        out_img_data[i, j, k, l] = tpc_odd[l/2]
    return out_img_data

# def complex_to_phase(img_data):
#     x = img_data.shape[0]
#     y = img_data.shape[1]
#     z = img_data.shape[2]
#     w = img_data.shape[3]
#     out_img_data = np.zeros(shape=(x, y, z, w))
#     for i in range(x):
#         for j in range(y):
#             for k in range(z):
#                 for l in range(w):
#                     out_img_data[i, j, k, l] = cmath.phase(img_data[i, j, k, l])
#     return out_img_data

# def complex_to_magnitude(img_data):
#     x = img_data.shape[0]
#     y = img_data.shape[1]
#     z = img_data.shape[2]
#     w = img_data.shape[3]
#     out_img_data = np.zeros(shape=(x, y, z, w))
#     for i in range(x):
#         for j in range(y):
#             for k in range(z):
#                 for l in range(w):
#                     out_img_data[i, j, k, l] = abs(img_data[i, j, k, l])
#     return out_img_data

def draw_phase_repartition(before, after):
    mag_before = [elem for elem in np.nditer(complex_to_magnitude(before))]
    phase_before = [elem for elem in np.nditer(complex_to_phase(before))]
    mag = [elem for elem in np.nditer(complex_to_magnitude(after))]
    phase = [elem for elem in np.nditer(complex_to_phase(after))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(phase_before, mag_before, alpha=0.75)
    ax.scatter(phase, mag, alpha=0.75)
    plt.show()

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
#complex_corrected_image = correct_phase_temporally(echotime, img_data, 3)
#draw_phase_repartition(to_complex(img_data), complex_corrected_image)

out_img_data = complex_to_magnitude(to_complex(img_data))
#out_img_data = complex_corrected_image.real
out_img = nib.Nifti1Image(out_img_data, np.eye(4))
out_img.to_filename(output)
