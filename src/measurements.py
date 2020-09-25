import numpy as np

def area_pixels(image):
    return np.count_nonzero(image)

def area_unit(image, resolution=(1,1,1)):
    return area_pixels(image) * resolution[0] * resolution[1] * resolution[2]

def average_value(image):
    return np.mean(image[image != 0])

def max_value(image):
    return image[image != 0].max() if image.any() else 0

def min_value(image):
    return image[image != 0].min() if image.any() else 0
