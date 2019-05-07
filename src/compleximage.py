import numpy as np
import math
import cmath

def separate_real_imaginary(img_data):
    real = np.zeros(shape=(img_data.shape[:-1]+ (img_data.shape[-1]//2,)))
    imaginary = np.zeros(shape=(img_data.shape[:-1]+ (img_data.shape[-1]//2,)))
    dim = len(img_data.shape)
    ri = img_data.shape[dim-1]+1
    for i, x in np.ndenumerate(img_data):
        image = img_data[i]
        if i[-1] < ri//2:
            real[i] = image
        else:
            index = i[:-1] + (i[-1] % (ri//2),)
            imaginary[index] = image
    return real, imaginary

def to_complex(img_data):
    complex_img = np.zeros(shape=(img_data.shape[:-1]+ (img_data.shape[-1]//2, )), dtype=complex)
    real, imaginary = separate_real_imaginary(img_data)
    for i, x in np.ndenumerate(complex_img):
        complex_img[i] = complex(real[i], imaginary[i])
    return complex_img

def complex_to_phase(img_data):
    out_img_data = np.zeros_like(img_data, dtype=float)
    for item, x in np.ndenumerate(img_data):
        out_img_data[item] = cmath.phase(img_data[item])
    return out_img_data

def complex_to_magnitude(img_data):
    out_img_data = np.zeros_like(img_data, dtype=float)
    for item, x in np.ndenumerate(img_data):
        out_img_data[item] = abs(img_data[item])
    return out_img_data
