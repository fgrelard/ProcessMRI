import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import threshold_otsu, rank
from skimage.draw import circle
from skimage.feature import canny
from skimage import measure
import cv2
import matplotlib.pyplot as plt


def detect_circle(image, threshold, min_radius, max_radius):
    cond = np.where(image < threshold)
    image_copy = np.copy(image)
    image_copy[cond] = 0
    edges = canny(image_copy, sigma=3, low_threshold=10, high_threshold=40)

    # Detect two radii
    hough_radii = np.arange(min_radius, max_radius, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    if len(cx) > 0:
        return cx[0], cy[0], radii[0]
    return -1, -1, -1



def detect_tube(image, threshold=150, min_radius=10, max_radius=50):
    cy, cx, radii = [], [], []
    for i in range(image.shape[0]):
        center_x, center_y, radius = detect_circle(image[i, :,:], threshold, min_radius, max_radius)
        if center_y >= 0:
            cy.append(center_y)
            cx.append(center_x)
            radii.append(radius)
    center_y = np.median(cy)
    center_x = np.median(cx)
    radius = np.median(radii)
    return center_x, center_y, radius

def detect_grain(image):
    threshold = threshold_otsu(image)
    cond = np.where(image > threshold)
    image_copy = np.zeros_like(image)
    image_copy[cond] = 255
    return image_copy

def properties_largest_area_cc(ccs):
    """
    Extracts the connected component
    with the largest area

    Parameters
    ----------
    ccs: numpy.ndarray
        connected components

    Returns
    ----------
    RegionProperties
        connected component with largest area

    """
    regionprops = measure.regionprops(ccs)
    if len(regionprops) == 0:
        return -1
    areas = lambda r: r.area
    argmax = max(regionprops, key=areas)
    return argmax

def region_property_to_cc(ccs, regionprop):
    """
    Extracts the connected component associated
    with the region

    Parameters
    ----------
    ccs: numpy.ndarray
        connected components
    regionprop: RegionProperties
        desired region

    Returns
    ----------
    numpy.ndarray
        the binary image (mask) of the desired region
    """
    label = regionprop.label
    cc = np.where(ccs == label, 255, 0)
    return cc

def largest_connected_component(image):
    ccs = measure.label(image, background=0)
    regionprop = properties_largest_area_cc(ccs)
    largest_cc = region_property_to_cc(ccs, regionprop)
    return largest_cc

def draw_defect(drawing, cnt, defects, r=2):
    if defects is None:
        return
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        cv2.circle(drawing,far,r,[255,0,0],-1)

def find_largest_contour(contours):
    max_perimeter = 0
    larg_cnt = None
    for i in range(len(contours)):
        contour = contours[i]
        perimeter = cv2.arcLength(contour, True)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            larg_cnt = contour
    return larg_cnt


def find_farthest_convexity_defect(cnt, defects):
    max_distance = 0
    point = None
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i, 0]
        if d > max_distance:
            point = tuple(cnt[f][0])
            max_distance = d
    return point

def find_location_cavity(image):
    blur = cv2.blur(image, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnt = find_largest_contour(contours)
    hull = cv2.convexHull(cnt, returnPoints=False)
    conv_defect = cv2.convexityDefects(cnt,hull)
    point = find_farthest_convexity_defect(cnt, conv_defect)
    return point

def detect_cavity(image):
    point = find_location_cavity(image)
    drawing = image.copy()
    cv2.circle(drawing, point, 2, (255, 0, 0), -1)
    plt.imshow(drawing)
    plt.show()
    return []
    # cond = np.where(image > 0)
    # threshold = threshold_otsu(image[cond])
    # cond = np.where(image > threshold)
    # image_copy = np.zeros_like(image)
    # image_copy[cond] = 255
    # return image_copy
