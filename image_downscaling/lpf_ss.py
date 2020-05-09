import cv2
import skimage.measure
import numpy as np

# Gaussian Filter Kernel Size 3

def gaussian_filter_3(img):
    result = cv2.GaussianBlur(img, (3, 3), 0)
    return result

# Gaussian Filter Kernel Size 5

def gaussian_filter_5(img):
    result = cv2.GaussianBlur(img, (5, 5), 0)
    return result

# Take median of four pixels

def sample_median(img):
    result = skimage.measure.block_reduce(img, (2,2,1), np.median)
    return result


def downscale_by_two(img):
    lpf_image = gaussian_filter_3(img)
    downscaled = sample_median(lpf_image)
    return downscaled