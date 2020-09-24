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
import warnings

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

def fit_linear_regression(x, y):
    """
    Fits a line by linear regression.

    Parameters
    ----------
    x: list
        x data
    y: list
        y data

    Returns
    ----------
    tuple
        linear coefficients, residuals
    """
    fit, residuals, rank, singular_values, rcond = np.polyfit(np.array(x), np.array(y), 1, full=True)
    values = [fit[0], fit[1], 0]
    linear_function = lambda x, a, b, c: a * x + b
    error = normalized_mse(values, x, y, fn=linear_function)
    return values, error

def fit_exponential_linear_regression(x, y, w=None):
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
    if w is None:
        w = [(1 - 1/len(x)*i)**2 for i in range(len(x))]
    fit, residuals, rank, singular_values, rcond = np.polyfit(np.array(x), np.log(y), 1,  w=w, full=True)
    exp_values = [np.exp(fit[1]), -fit[0], 0]
    error = normalized_mse(exp_values, x, y)
    return exp_values, error


def piecewise_linear(x, x0, y0, k1, k2):
    """
    Piecewise function involving
    two linear functions.

    Parameters
    ----------
    x: list
        x data
    x0: float
        x value where the two linear functions intersect
    y0: float
        y value where the two linear functions intersect
    k1: float
        slope for first linear function
    k2: float
        slope fot second linear function

    Returns
    ----------
    np.piecewise
        the piecewise function
    """
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


def fit_exponential_piecewise_linear_regression(x, y):
    """
    Exponential fitting by piecewise linear regression.
    Involves two linear functions fitted on the logarithm
    of y. Particularly suited to bi-exponential distributions.

    Parameters
    ----------
    x: list
        x data
    y: list
        y data

    Returns
    ----------
    tuple
        exponential coefficients, residuals
    """
    fit, _ = fit_exponential_linear_regression(x, y)
    guesses = [-fit[1], np.log(fit[0])]
    guess_x0 = len(x) // 2
    guess_k1 = -guesses[0]
    guess_y0 = guesses[1] + guess_k1 * guess_x0
    guess_k2 = guess_k1 * 2
    popt, pcov = curve_fit(piecewise_linear, x, np.log(y), p0=(guess_x0, guess_y0, guess_k1, guess_k2))
    a = popt[2]
    c = popt[3]
    b = popt[1] - a * popt[0]
    d = popt[1] - c * popt[0]
    intersect_one = np.exp(b)
    intersect_two = np.exp(d)
    exp_values = [intersect_one, np.abs(a), 0]
    error = normalized_mse(exp_values, x, y)
    return exp_values, error


def normalized_mse(exp_values, x, y, fn=n_exponential_function):
    """
    Normalized Mean Squared Error (MSE).
    Defined as the MSE between a fitted distribution (exp_values)
    and real data (y), where both distribution are normalized
    with respect to the maximum value in either y or the
    fitted distribution.

    Parameters
    ----------
    exp_values: list
        exponential coefficients
    x: list
        x data
    y: list
        y data
    fn: function
        function to apply to get fitted distribution from exp_values

    Returns
    ----------
    float
        normalized MSE
    """
    fitted = fn(np.array(x), *exp_values)
    fitted_norm = fitted * 1.0 / max(fitted.max(), y.max())
    y_norm = y * 1.0 / max(fitted.max(), y.max())
    error = np.sqrt((fitted_norm - y_norm)**2)
    return np.mean(error)



def fit_exponential(x, y, p0, lreg=False, biexp=False, piecewise_lreg=False):
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
    biexp: bool
        whether linear regression should be adapted to bi-
        exponential functions
    piecewise_lreg: bool
        whether linear regression should be piecewise (two
        linear functions are fitted on the log of the data)
    """
    initial_values = [y[0], float("inf"), 0]
    if lreg:
        fit_lr, residual_lr = fit_linear_regression(x, y)
        fit_elr, residual_elr = fit_exponential_linear_regression(x, y)
        # if a line is more suited than an exponential function,
        # we use another way of fitting the exponential
        # (different weights)
        if residual_lr < residual_elr or not biexp:
            fit, residual = fit_exponential_linear_regression(x, y, w=np.sqrt(y))
        else:
            fit, residual = fit_elr, residual_elr
    elif piecewise_lreg:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                fit, residual = fit_exponential_piecewise_linear_regression(x, y)
            except RuntimeError as e:
                fit, residual = fit_exponential_linear_regression(x, y)
                fit = [0, 1, 0]
                if len(fit) != len(p0):
                    diff = len(p0) - len(fit)
                    fit += [0 for i in range(diff)]
    else:
        try:
            fit, pcov = curve_fit(n_exponential_function, x, y, p0=p0,maxfev=3000)
            residual = normalized_mse(fit, x, y)
            if fit[1] > 3:
                raise RuntimeError("Exponential coefficient not suited.")
        except RuntimeError as e:
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
