import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import threshold_otsu, rank
from skimage.draw import circle
from skimage.feature import canny
from skimage import measure
from skimage.util import img_as_ubyte

import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage as ndi
import SimpleITK as sitk

def detect_circle(image, threshold, min_radius, max_radius):
    cond = np.where(image < threshold)
    image_copy = np.copy(image)
    image_copy[cond] = 0
    edges = canny(image_copy, sigma=3, low_threshold=10, high_threshold=30)
    # Detect two radii
    hough_radii = np.arange(min_radius, max_radius, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    if len(cx) > 0:
        return cx[0], cy[0], radii[0]
    return -1, -1, -1

def median_circle(image):
    depth = image.shape[0]
    L = np.zeros(shape=(depth, 3))
    for i in range(depth):
        image_current = image[i, ...]
        image_current = img_as_ubyte(image_current * 1.0 / image_current.max())
        threshold = threshold_otsu(image_current)
        cx, cy, r = detect_circle(image_current, threshold,10,20)
        L[i, 0] = cx
        L[i, 1] = cy
        L[i, 2] = r
        circx, circy = circle(cx, cy, r, shape=image_current.shape)
        image_current[circy, circx] = 0
    cx, cy, r = np.median(L, axis=0)
    return cx, cy, r

def remove_circle(image, cx, cy, r):
    circx, circy = circle(cx, cy, r, shape=image[0,...].shape)
    image[..., circy, circx] = 0
    return image


def binarize(image):
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
    ret, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    cnt = find_largest_contour(contours)
    hull = cv2.convexHull(cnt, returnPoints=False)
    conv_defect = cv2.convexityDefects(cnt,hull)
    point = find_farthest_convexity_defect(cnt, conv_defect)
    return point



def compute_pointness(I, n=5):
    # Compute gradients
    # GX = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=5, scale=1)
    # GY = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=5, scale=1)
    GX = cv2.Scharr(I, cv2.CV_32F, 1, 0, scale=1)
    GY = cv2.Scharr(I, cv2.CV_32F, 0, 1, scale=1)
    GX = GX + 0.0001  # Avoid div by zero

    # Threshold and invert image for finding contours
    _, I = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY)
    # Pass in copy of image because findContours apparently modifies input.
    C, H = cv2.findContours(I.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    heatmap = np.zeros_like(I, dtype=np.float)
    pointed_points = []
    for contour in C:
        contour = contour.squeeze()
        measure = []
        N = len(contour)
        for i in range(N):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + n) % N]

            # Angle between gradient vectors (gx1, gy1) and (gx2, gy2)
            gx1 = GX[y1, x1]
            gy1 = GY[y1, x1]
            gx2 = GX[y2, x2]
            gy2 = GY[y2, x2]
            cos_angle = gx1 * gx2 + gy1 * gy2
            cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
            angle = np.arccos(cos_angle)
            if cos_angle < 0:
                angle = np.pi - angle

            x1, y1 = contour[((2*i + n) // 2) % N]  # Get the middle point between i and (i + n)
            heatmap[y1, x1] = angle  # Use angle between gradient vectors as score
            measure.append((angle, x1, y1, gx1, gy1))

        _, x1, y1, gx1, gy1 = max(measure)  # Most pointed point for each contour

        # Possible to filter for those blobs with measure > val in heatmap instead.
        pointed_points.append((x1, y1, gx1, gy1))

    heatmap = cv2.GaussianBlur(heatmap, (3, 3), heatmap.max())
    return heatmap, pointed_points


def plot_points(image, pointed_points, radius=5, color=(255, 0, 0)):
    for (x1, y1, _, _) in pointed_points:
        cv2.circle(image, (x1, y1), radius, color, -1)

def line_to_vector(line):
    p1 = line[0]
    p2 = line[1]
    gx = p2[0] - p1[0]
    gy = p2[1] - p1[1]
    return gx, gy

def angle_line(gx1, gx2, gy1, gy2):
    cos_angle = gx1 * gx2 + gy1 * gy2
    cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
    angle = np.arccos(cos_angle)
    # if cos_angle < 0:
    #     angle = np.pi - angle
    return angle * 180.0 / np.pi
    #return angle
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180.0 / np.pi

def compute_curvature(image):
    blur = cv2.blur(image, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnt = find_largest_contour(contours)
    epsilon = 0.001*cv2.arcLength(cnt,True)
    poly = cv2.approxPolyDP(cnt, epsilon, True)
    lines = []
    heatmap = np.zeros_like(image, dtype=np.float)

    for i in range(len(poly)-1):
        p_i = poly[i][0]
        p_next = poly[i+1][0]
        line = [p_i, p_next]
        lines.append(line)
    lines.append([poly[-1][0], poly[0][0]])
    for i in range(len(lines)-1):
        first_line = lines[i]
        second_line = lines[i+1]
        point = lines[i][1]
        gx1, gy1 = line_to_vector([first_line[1], first_line[0]])
        gx2, gy2 = line_to_vector(second_line)
        value = angle_line(gx1, gx2, gy1, gy2)
        heatmap[point[1], point[0]] = value
    return heatmap

def neighbors(im, p, d=1):
    i = p[1]
    j = p[0]
    n = im[i-d:i+d+1, j-d:j+d+1].copy()
    if n.size != 9:
        return None
    return n

def find_local_maximum_dt(dt, point, r=1):
    new_point = point
    index = (1,1)
    is_max = True
    while index != (0,0) and is_max:
        n = neighbors(dt, new_point, r)
        if n is None:
            break
        center_value = n[r, r]
        n[r, r] = 0
        max_dt = np.amax(n)
        index = np.unravel_index(np.argmax(n.T, axis=None), n.shape)
        index = tuple(x - r for x in index)
        is_max = (max_dt > center_value)
        if is_max:
            new_point = tuple(x + y for x, y in zip(new_point, index))
    new_point = tuple(int(x) for x in new_point)
    return new_point

def detect_cavity(image):
    point = find_location_cavity(image)
    cond = np.where(image > 0)
    threshold = threshold_otsu(image[cond])
    cond = np.where(image > threshold)
    image_copy = np.zeros_like(image)
    image_copy[cond] = 255
    distance = ndi.distance_transform_edt(image_copy)
    distance =  cv2.blur(distance, (3, 3)) # blur the image
    seed = find_local_maximum_dt(distance, point)
    seg = sitk.ConfidenceConnected(sitk.GetImageFromArray(distance), seedList=[seed], numberOfIterations=1, multiplier=2.5, initialNeighborhoodRadius=1, replaceValue=255)
    seg_array = sitk.GetArrayFromImage(seg)
    #seg_array = ndi.morphology.binary_fill_holes(seg_array)
    return seg_array


def detect_grain_3D(image):
    image_copy = image.copy()
    depth = image.shape[0]
    image8 = img_as_ubyte(image_copy * 1.0 / image_copy.max())
    for i in range(depth):
        binarized = binarize(image8[i, ...])
        grain = largest_connected_component(binarized)
        cond = (i, ) + np.where(grain == 0)
        image_copy[cond] = 0
    return image_copy


def detect_cavity_3D(image):
    image_copy = image.copy()
    depth = image.shape[0]
    image8 = img_as_ubyte(image * 1.0 / image.max())
    for i in range(depth):
        cavity = detect_cavity(image8[i, ...])
        cond = (i, ) + np.where(cavity == 0)
        image_copy[cond] = 0
    return image_copy
