import numpy as np
from math import exp
from scipy.optimize import curve_fit

# Exponential function to be fitted against the data
def exponential_function(x, a, b, c):
    return a*np.exp(-b*x)+c

# Fit the exponential on the (x, y) data
def fit_exponential(x, y):
    try:
        popt, pcov = curve_fit(exponential_function, x[::2], y[::2])
        return popt[0] + popt[2]
    except RuntimeError:
        fit_lreg = fit_exponential_linear_regression(x, y)
        return fit_lreg

def fit_exponential_linear_regression(x, y):
    fit = np.polyfit(np.array(x), np.log(y), 1, w=np.sqrt(y))
    return np.exp(fit[1])


# Main function to estimate the parametric image
def estimation_density_image(echotime, image, threshold):
    x = image.shape[0]
    y = image.shape[1]
    data = np.zeros(shape=(x, y))
    for i in range(x):
        for j in range(y):
            pixel_values = image[i, j, :]
            if (pixel_values[0]) > threshold:
                pixel_value = fit_exponential(echotime, pixel_values)
                data[i, j] = pixel_value
            else:
                data[i, j] = pixel_values[0]
    return data
