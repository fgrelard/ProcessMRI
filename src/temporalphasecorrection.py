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

def correct_phase_temporally(echotimes, img_data, order, noise=0):
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

    magnitude_img = ci.complex_to_magnitude(complex_img_data)

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
        tpc_even = correct_phase_1d(even_echotime, tpc_even, order)
        tpc_odd = correct_phase_1d(odd_echotime, odd_complex_img[index], order)
        tpc_odd = correct_phase_1d(odd_echotime, tpc_odd, order)
        for k in range(out_img_data.shape[-1]):
            pointwise_index = index + (k, )
            value = magnitude_img[pointwise_index]
            if value < noise:
                out_img_data[pointwise_index] = complex_img_data[pointwise_index]
            elif k % 2 == 0:
                out_img_data[pointwise_index] = tpc_even[k//2]
            else:
                out_img_data[pointwise_index] = tpc_odd[k//2]
    return out_img_data


def draw_phase_repartition(before, after, noise):
    """
    Draws repartition of real and imaginary components

    Parameters
    ----------
    before: numpy.ndarray
        n-D complex non-corrected image
    after: numpy.ndarray
        n-D complex corrected image


    """
    mag_before = ci.complex_to_magnitude(before)
    mag_after =  ci.complex_to_magnitude(after)

    phase_before =  ci.complex_to_phase(before)
    phase_after =  ci.complex_to_phase(after)

    plot_mag_before = []
    plot_mag_after = []
    plot_phase_before = []
    plot_phase_after = []

    for index in np.ndindex(mag_before.shape):
        mag_before_value = mag_before[index]
        mag_after_value = mag_after[index]

        phase_before_value = phase_before[index]
        phase_after_value = phase_after[index]

        if (mag_before_value > noise):
            plot_mag_before.append(mag_before_value)
            plot_phase_before.append(phase_before_value)

        if (mag_after_value > noise):
            plot_mag_after.append(mag_after_value)
            plot_phase_after.append(phase_after_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(plot_phase_before, plot_mag_before, alpha=0.75)
    ax.scatter(plot_phase_after, plot_mag_after, alpha=0.75)
    plt.show()
