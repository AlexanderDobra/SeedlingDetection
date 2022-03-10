import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import cv2

def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return minmax_custom(arr, arr_min, arr_max)

def minmax_custom(arr, min, max):
    return (arr - min) / ((max - min) + 1e-6)

def minmax_over_nonzero(arr):
    arr_min = np.min(arr[np.nonzero(arr)])
    arr_max = np.max(arr)
    return minmax_custom(arr, arr_min, arr_max)

def minmax_reverse(arr, min, max):
    return arr * (max - min) + min

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging over nonzero elements."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    arr2 = arr.reshape(shape)
    cond = (arr2 > 0).sum(axis=(1, 3))
    out = np.zeros(new_shape)
    np.true_divide(arr2.sum(axis=(1, 3)), cond, where=(cond) > 0, out=out)
    return out

def interpolate_on_missing(arr, equal_to=0, method='nearest'):
    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    # mask invalid values
    array = np.ma.masked_equal(arr, equal_to)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    res = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method, fill_value=0)
    res = np.clip(res, 0, 1)
    return res

def get_edges(image, dim, treshold=0.7, visualise=False, read_from_file=False):
    if read_from_file: image = cv2.imread(image)
    image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    edges = cv2.Laplacian(image, cv2.CV_16S, ksize=5)
    cv2.convertScaleAbs(edges, edges)
    edges = minmax(edges)
    zeros = np.zeros(edges.shape)
    ones = np.ones(edges.shape)
    edges = np.where(edges < treshold, zeros, ones)
    if visualise:
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    return edges