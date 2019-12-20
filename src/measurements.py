import numpy as np

def area_pixels(image):
    return np.count_nonzero(image)

def area_unit(image, resolution=(1,1,1)):
    return area_pixels(image) * resolution[0] * resolution[1] * resolution[2]

def average_value(image):
    return np.mean(image)

def max_value(image):
    return image.max()

def min_value(image):
    return image.min()
