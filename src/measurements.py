import numpy as np

def area_pixels(image):
    return np.count_nonzero(image)

def area_unit(image, resolution=(1,1,1)):
    old_shape = image.shape
    new_shape = tuple(map(lambda a, b: a*b, old_shape, resolution))
    scaled_image = np.resize(image, new_shape)
    return area_pixels(scaled_image)

def average_value(image):
    return np.mean(image)

def max_value(image):
    return image.max()

def min_value(image):
    return image.min()
