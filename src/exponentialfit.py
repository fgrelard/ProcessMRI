import numpy as np
from math import exp
from scipy.optimize import curve_fit
import sklearn.mixture as skm

# Exponential function to be fitted against the data
def exponential_function(x, a, b, c):
    return a*np.exp(-b*x)+c

# Fit the exponential on the (x, y) data
def fit_exponential(x, y):
    try:
        popt, pcov = curve_fit(exponential_function, x, y)
        return popt[0] + popt[2]
    except RuntimeError:
        fit_lreg = fit_exponential_linear_regression(x, y)
        return fit_lreg

def fit_exponential_linear_regression(x, y):
    fit = np.polyfit(np.array(x), np.log(y), 1, w=np.sqrt(y))
    return np.exp(fit[1])


def auto_threshold(data):
    gmm = skm.GaussianMixture(n_components=2)
    gmm = gmm.fit(data)
    print(gmm.score_samples)

# Main function to estimate the parametric image
def estimation_density_image(echotime, image, threshold):
    data = np.zeros(shape=image.shape[:-1])
    auto_threshold(image)
    for i in np.ndindex(data.shape):
        pixel_values = image[i + (slice(None),)]
        if (pixel_values[0]) > threshold:
            pixel_value = fit_exponential(echotime, pixel_values)
            data[i] = pixel_value
        else:
            data[i] = pixel_values[0]
    return data
