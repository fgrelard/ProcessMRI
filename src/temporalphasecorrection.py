import numpy as np
import math
import matplotlib.pyplot as plt
import skimage
import cmath
import skimage.restoration as skires
import src.compleximage as ci

def unwrap_phases(phase_image):
    """
    Unwrap the phase modulo 2pi
    so that the difference between two consecutive
    phases is no more than pi

    Parameters
    ----------
    phase_image: numpy.ndarray
        n-D phase image

    Returns
    ----------
    numpy.ndarray
        n-D unwrapped phase image

    """
    unwrapped_image = np.zeros_like(phase_image, dtype=float)
    for index in np.ndindex(phase_image.shape[2:]):
        nd_index = (slice(None),slice(None))+index
        unwrap_2D = skires.unwrap_phase(skimage.img_as_float(phase_image[nd_index]))
        unwrapped_image[nd_index] = unwrap_2D
    return unwrapped_image


def correct_phase_1d(echotimes, decays, order, phases_unwrap=None):
    """
    Corrects the phase for 1D-decays

    Parameters
    ----------
    echotimes: list
        x data
    decays: list
        complex data
    order: int
        polynomial order to fit on the unwrapped phase
    phases_unwrap: numpy.ndarray
        n-D unwrapped phase image
        (default: None, uses numpy.unwrap function)

    Returns
    ----------
    list
        corrected complex data

    """
    if phases_unwrap is None:
        phases = [cmath.phase(decay) for decay in decays]
        phases_unwrap = np.unwrap(phases)

    weighting_vector = np.ones(shape=(len(phases_unwrap)))
    p = np.polyfit(echotimes, phases_unwrap, order, w=weighting_vector)
    f = np.polyval(p, echotimes)
    decay_new = np.multiply(decays, np.exp(-1j*np.transpose(f)))
    return decay_new

def correct_phase_temporally(echotimes, img_data, order):
    """
    Corrects the phase temporally (main function)
    on n-D complex images

    Parameters
    ----------
    echotimes: list
        x data
    img_data: numpy.ndarray
        n-D complex image
    order: int
        polynomial order

    Returns
    ----------
    type
        description

    """
    out_img_data = np.zeros(shape=(img_data.shape[:-1]+ (img_data.shape[-1]//2, )), dtype=complex)
    ri = img_data.shape[-1]
    complex_img_data = ci.to_complex(img_data)
    even_echotime = echotimes[:ri//2:2]
    odd_echotime = echotimes[1:ri//2:2]

    #Separating even and odd echoes
    even_complex_img = complex_img_data[..., ::2]
    odd_complex_img = complex_img_data[..., 1::2]
    phase_image = ci.complex_to_phase(complex_img_data)
    phases_unwrapped = unwrap_phases(phase_image)

    #Iterating over the even and odd slices
    for index in np.ndindex(even_complex_img.shape[:-1]):
        phase_unwrapped_even = phases_unwrapped[index + (slice(None, None, 2),)]
        phase_unwrapped_odd = phases_unwrapped[index + (slice(1, None, 2),)]
        tpc_even = correct_phase_1d(even_echotime, even_complex_img[index], order)
        tpc_odd = correct_phase_1d(odd_echotime, odd_complex_img[index], order)
        for k in range(out_img_data.shape[-1]):
            pointwise_index = index + (k, )
            if k % 2 == 0:
                out_img_data[pointwise_index] = tpc_even[k//2]
            else:
                out_img_data[pointwise_index] = tpc_odd[k//2]
    return out_img_data


def draw_phase_repartition(before, after):
    """
    Draws repartition of real and imaginary components

    Parameters
    ----------
    before: numpy.ndarray
        n-D complex non-corrected image
    after: numpy.ndarray
        n-D complex corrected image


    """
    mag_before = [elem for elem in np.nditer(ci.complex_to_magnitude(before))]
    phase_before = [elem for elem in np.nditer(ci.complex_to_phase(before))]
    mag = [elem for elem in np.nditer(ci.complex_to_magnitude(after))]
    phase = [elem for elem in np.nditer(ci.complex_to_phase(after))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(phase_before, mag_before, alpha=0.75)
    ax.scatter(phase, mag, alpha=0.75)
    plt.show()
