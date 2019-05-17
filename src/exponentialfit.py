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

def denoise_image(image):
    sigma_est = np.mean(skrestore.estimate_sigma(image, multichannel=True))
    print(sigma_est)
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    denoise2_fast = skrestore.denoise_nl_means(image, h=1.5*sigma_est,fast_mode=False, **patch_kw)
    return denoise2_fast

# Exponential function to be fitted against the data
def exponential_function(x, a, b):
    return a*np.exp(-b*x)

def n_exponential_function(x, *params):
    n = (len(params)-1)//2
    exp = 0
    for i in range(n):
        exp += exponential_function(x, params[i*2], params[i*2+1])
    return exp + params[-1]

def biexponential_function(x, a, b, c, d, e):
    return a*np.exp(-b*x) + d*np.exp(-e*x) + c


def density(values):
    return np.sum(values[0:-1:2]) + values[-1]

def t2_star(values, echotime):
    return np.sum(np.divide(echotime,values[1::2]))

def fit_exponential_linear_regression(x, y):
    fit = np.polyfit(np.array(x), np.log(y), 1,  w=np.sqrt(y))
    return [np.exp(fit[1]), -fit[0], 0]

# Fit the exponential on the (x, y) data
def fit_exponential(x, y, p0, lreg=False):
    if lreg:
        return fit_exponential_linear_regression(x, y)
    try:
        popt, pcov = curve_fit(n_exponential_function, x, y, p0=p0,maxfev=3000)
        # poptbi, pcovbi = curve_fit(biexponential_function, x, y)
        # popt_odd, pcov_odd = curve_fit(n_exponential_function, x[1::2], y[1::2], p0=p0, maxfev=3000)
        # popt_even, pov_even = curve_fit(n_exponential_function, x[::2], y[::2], p0=p0, maxfev=3000)
        # if popt_odd[1] < popt[1] and popt_odd[1] < popt_even[1]:  #
        #     popt = popt_odd
        # elif popt_even[1] < popt[1] and popt_even[1] < popt_odd[1]: #
        #     popt = popt_even

        if popt[1] > 3:
            raise RuntimeError("Exponential coefficient not suited.")
        return popt
    except RuntimeError as error:
        return fit_exponential_linear_regression(x, y)

def plot_values(x, y, value, popt, threshold, f=n_exponential_function):
    if value > threshold:
        xx = np.linspace(0, 9, 1000)
        yy = f(xx, *popt)
        plt.plot(x, y, '.', xx, yy)
        plt.show()

def auto_threshold_gmm(data, number_gaussian):
    gmm = skm.GaussianMixture(n_components=number_gaussian, max_iter=250)
    gmm = gmm.fit(data)

    mean = gmm.means_
    covariance = gmm.covariances_.ravel()

    argmax_mean = np.argmax(mean)
    return mean[argmax_mean] - np.sqrt(covariance[argmax_mean])

def compute_edges(image):
    edges_sobel = np.zeros_like(image)
    for index in np.ndindex(image.shape[2:]):
        index_xy = (slice(None), slice(None)) + index
        e = skfilters.sobel(image[index_xy])
        edges_sobel[index_xy] = np.interp(e, (e.min(), e.max()), (0, 255))
    return edges_sobel

def n_to_p0(n, y0):
    p0 = ()
    for i in range(n):
        p0 += (2*y0, 0.1)
    p0 += (0,)

# Main function to estimate the parametric image
def estimation_density_image(echotime, image,threshold=None, lreg=True, n=1):
    data = np.zeros(shape=image.shape[:-1])

    #Auto threshold with mixture of gaussian (EM alg.)
    if threshold is None:
        threshold = auto_threshold_gmm(np.expand_dims(image[...,0].ravel(), 1), 3)

    for i in np.ndindex(data.shape):
        pixel_values = image[i + (slice(None),)]
        if pixel_values[0] > threshold:
            p0 = n_to_p0(n, pixel_values[0])
            fit = fit_exponential(echotime, pixel_values, p0, lreg)
            pixel_value = density(fit)
            data[i] = pixel_value
        else:
            data[i] = pixel_values[0]
    return data
