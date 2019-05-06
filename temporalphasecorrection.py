import numpy as np
import math
import json
import matplotlib.pyplot as plt
import skimage
import cmath
from skimage.restoration import unwrap_phase

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
