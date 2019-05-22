import numpy as np
import math
import cmath

def separate_real_imaginary(img_data):
    """
    Separates the real and imaginary parts
    in a complex image.
    Real and imaginary parts are assumed to be
    mixed in the last dimension of size n
    With real contained in the first n//2 slices
    and imaginary in the last n//2 slices

    Parameters
    ----------
    img_data: numpy.ndarray
        n-D complex image

    Returns
    ----------
    real: numpy.ndarray
        real image

    imaginary: numpy.ndarray
        imaginary image
    """
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
    """
    Converts a complex image where real and
    imaginary are separated into a single complex
    image, with pixel value = re+i*im

    Parameters
    ----------
    img_data: numpy.ndarray
        n-D complex image with real and imaginar
        parts separated

    Returns
    ----------
    numpy.ndarray
        n-D complex image with complex values

    """
    complex_img = np.zeros(shape=(img_data.shape[:-1]+ (img_data.shape[-1]//2, )), dtype=complex)
    real, imaginary = separate_real_imaginary(img_data)
    for i, x in np.ndenumerate(complex_img):
        complex_img[i] = complex(real[i], imaginary[i])
    return complex_img

def complex_to_phase(img_data):
    """
    Converts a complex image with complex values
    to the phase image (atan2(im,re))

    Parameters
    ----------
    img_data: numpy.ndarray
        n-D complex image with complex values

    Returns
    ----------
    numpy.ndarray
        n-D phase image with float values
    """
    out_img_data = np.zeros_like(img_data, dtype=float)
    for item, x in np.ndenumerate(img_data):
        out_img_data[item] = cmath.phase(img_data[item])
    return out_img_data

def complex_to_magnitude(img_data):
    """
    Converts a complex image with complex values
    to the magnitude image (sqrt(re^2+im^2))

    Parameters
    ----------
    img_data: numpy.ndarray
        n-D complex image with complex values

    Returns
    ----------
    numpy.ndarray
        n-D magnitude image with float values
    """
    out_img_data = np.zeros_like(img_data, dtype=float)
    for item, x in np.ndenumerate(img_data):
        out_img_data[item] = abs(img_data[item])
    return out_img_data
