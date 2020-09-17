import numpy as np
from math import exp
from scipy.optimize import curve_fit
import sklearn.mixture as skm
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.restoration as skrestore
import skimage.filters as skfilters
import skimage
import os
import time

def denoise_image(image, size, distance, spread):
    """
    Image denoising by non-local means
    Adapted to Rician noise
    For up to 4D images

    Parameters
    ----------
    image: numpy.ndarray
        description
    size: int
        patch size
    distance: int
        patch distance search
    spread: float
        factor by which the estimated noise variance is multiplied for noise correction

    Returns
    ----------
    numpy.ndarray:
        the denoised image

    """
    denoised = np.zeros_like(image)
    dim = len(image.shape)
    if dim > 3:
        for i in range(image.shape[-1]):
            image3D = denoise_2_3D(image[...,i], size, distance, spread)
            denoised[..., i] = image3D
    else:
        denoised = denoise_2_3D(image, size, distance, spread)
    return denoised

def denoise_2_3D(image, size, distance, spread):
    """
    Image denoising by non-local means
    Adapted to Rician noise
    For 2D and 3D images

    Parameters
    ----------
    image: numpy.ndarray
        description
    size: int
        patch size
    distance: int
        patch distance search
    spread: float
        factor by which the estimated noise variance is multiplied for noise correction

    Returns
    ----------
    numpy.ndarray:
        the denoised image

    """
    sigma_est = np.mean(skrestore.estimate_sigma(image, multichannel=True))
    patch_kw = dict(patch_size=size,      # s*s patches
                patch_distance=distance,  # d*d search area
                multichannel=True)
    image = image.copy(order='C')
    denoised = skrestore.denoise_nl_means(image, h=spread*sigma_est,fast_mode=False, **patch_kw)
    return denoised

def exponential_function(x, a, b):
    """
    Mono-exponential function : ae-(bx)

    Parameters
    ----------
    x: float
        data
    a: float
        first parameter
    b: float
        second parameter

    Returns
    ----------
    float:
        the exponential of x

    """
    return a*np.exp(-b*x)

def n_exponential_function(x, *params):
    """
    n-exponential function

    Parameters
    ----------
    x: np.ndarray
        data
    params: list
        number of parameters, determining the order of the exponential

    Returns
    ----------
    numpy.ndarray:
        n-exponential of x

    """
    n = (len(params)-1)//2
    exp = 0
    for i in range(n):
        exp += exponential_function(x, params[i*2], params[i*2+1])
    return exp + params[-1]


def density(values):
    """
    Density from exponential coefficients

    Parameters
    ----------
    values: list
        list of exponential coefficients

    Returns
    ----------
    float
        the density
    """
    return np.sum(values[0:-1:2]) + values[-1]

def t2_star(values, echotime):
    """
    T2-star from exponential coefficients
    Parameters
    ----------
    values: list
        list of exponential coefficients
    echotime:
        echotime spacing

    Returns
    ----------
    float
        t2 star

    """
    vals = [x for x in values[1::2] if x > 0]
    t2_val = np.sum(np.divide(echotime,vals))
    return t2_val if t2_val > 0 else 0

def fit_exponential_linear_regression(x, y):
    """
    Fit a mono-exponential by linear regression
    on the log of the data

    Parameters
    ----------
    x: list
        x data
    y: numpy.array
        y data

    Returns
    ----------
    tuple
        exponential coefficients, residuals
    """
    fit, residuals, rank, singular_values, rcond = np.polyfit(np.array(x), np.log(y), 1,  w=[1 for i in range(len(y))], full=True)
    exp_values = [np.exp(fit[1]), -fit[0], 0]
    error = normalized_mse(exp_values, x, y)
    return exp_values, error

def normalized_mse(exp_values, x, y):
    fitted = n_exponential_function(np.array(x), *exp_values)
    fitted_norm = fitted * 1.0 / max(fitted.max(), y.max())
    y_norm = y * 1.0 / max(fitted.max(), y.max())
    error = np.sqrt((fitted_norm - y_norm)**2)
    return np.mean(error)

def fit_exponential(x, y, p0, lreg=False):
    """
    Fit the exponential on the (x, y) data
    Parameters
    ----------
    x: list
        x data
    y: numpy.array
        y data
    p0: tuple
        first approximation of exponential coefficients
        the number of arguments in the tuple determines the order
        of the exponential
    lreg: bool
        use linear regression or nnls
    """
    initial_values = [y[0], float("inf"), 0]
    if lreg:
        fit, residual = fit_exponential_linear_regression(x, y)
        return fit, residual
    try:
        popt, pcov = curve_fit(n_exponential_function, x, y, p0=p0,maxfev=3000)
        residual = normalized_mse(popt, x, y)
        if popt[1] > 3:
            raise RuntimeError("Exponential coefficient not suited.")
        return popt, residual
    except RuntimeError as error:
        fit, residual = fit_exponential_linear_regression(x, y)
        if len(fit) != len(p0):
            diff = len(p0) - len(fit)
            fit += [0 for i in range(diff)]
        return fit, residual

def plot_values(x, y, value, popt, threshold, f=n_exponential_function):
    """
    Plot values and corresponding exponential fit

    Parameters
    ----------
    x: list
        x data
    y: numpy.array
        y data
    value: float
        value to check
    popt: list
        exponential coefficients from fit
    threshold: int
        threshold to check value against
    f: function
        function to use to use the exponential coefficients

    """
    if value > threshold:
        xx = np.linspace(0, 9, 1000)
        yy = f(xx, *popt)
        plt.plot(x, y, '.', xx, yy)
        plt.xlabel("Temps d'échos")
        plt.ylabel("Intensités")
        plt.show()

def auto_threshold_gmm(data, number_gaussian):
    """
    Auto thresholding through fitting of a mixture of gaussians
    by the expectation maximization algorithm

    Parameters
    ----------
    data: image
        numpy.ndarray
    number_gaussian: int
        number of gaussians to fit

    Returns
    ----------
    int
        threshold as mu_0 + 2 * sigma_0
    """
    gmm = skm.GaussianMixture(n_components=number_gaussian, max_iter=250)
    gmm = gmm.fit(data)

    mean = gmm.means_
    covariance = gmm.covariances_.ravel()

    argmin_mean = np.argmin(mean)
    return mean[argmin_mean] + 2*np.sqrt(covariance[argmin_mean])

def n_to_p0(n, y0):
    """
    Construct exponential coefficients from order

    Parameters
    ----------
    n: int
        order
    y0: float
        first y-point

    Returns
    ----------
    tuple
        the exponential coefficients

    """
    p0 = ()
    for i in range(n):
        p0 += (2*y0, 0.1)
    p0 += (0,)
    return p0

# Main function to estimate the parametric image
def exponentialfit_image(echotime, image,threshold=None, lreg=True, n=1):
    """
    Exponential fit on an image

    Parameters
    ----------
    echotime: list
        x data
    image: numpy.ndarray
        image with multiple echoes in last dimension
    threshold: float
        value below which exponential fit is not performed
        (snr low)
    lreg: bool
        linear regression or nnls
    n: int
        order of exponential

    Returns
    ----------
    desnity_data: numpy.ndarray
        density image

    t2_data: numpy.ndarray
        t2* image
    """
    density_data = np.zeros(shape=image.shape[:-1])
    t2_data = np.zeros(shape=image.shape[:-1])

    #Auto threshold with mixture of gaussian (EM alg.)
    if threshold is None:
        threshold = auto_threshold_gmm(np.expand_dims(image[...,0].ravel(), 1), 3)

    for i in np.ndindex(density_data.shape):
        pixel_values = image[i + (slice(None),)]
        if pixel_values[0] > threshold:
            p0 = n_to_p0(n, pixel_values[0])
            fit = fit_exponential(echotime, pixel_values, p0, lreg)
            density_value = density(fit)
            t2_value = t2_star(fit, echotime[0])

            density_data[i] = density_value
            t2_data[i] = t2_value
        else:
            density_data[i] = pixel_values[0]
            t2_data[i] = 0
    return density_data, t2_data
