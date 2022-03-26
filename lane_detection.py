"""Detect straight lanes of a road using Hough transforms."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edges(image: np.ndarray) -> np.ndarray:
    """Detect the edges of an image using Canny edge detector.

    Args:
        image: Image whose edges we want to detect.

    Returns:
        Image with detected edges.
    """

    canny_image = cv2.Canny(image, 50, 150)

    return canny_image


def road_mask(image: np.ndarray) -> np.ndarray:
    """Generate mask that is used to filter edges just from the road.

    Args:
        image: Canny image containing the edges of the original image.

    Returns:
        Image mask where our wanted region is white.
    """
    shape = image.shape
    black_mask = np.zeros_like(image)
    triangle = np.array([[(200, shape[0]), (1100, shape[0]), (550, 200)]])

    mask = cv2.fillPoly(black_mask, triangle, 255)
    return mask


def road_lines(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """USe canny edge image and mask to find edges in ROI.

    Args:
        image: Canny image containing the edges of the original image.
        mask: Image containing the mask of the ROI.

    Returns:
        Image containing the edges within ROI.
    """
    roi_edges = image & mask
    return roi_edges


# Load image
image = cv2.imread("test_image.jpg")
color_image = image.copy()

# Convert image to grayscale (requires less computations because of less channels)
gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# Canny image
canny_image = canny_edges(gray)

# Masked image
mask_image = road_mask(canny_image)

# ROI edges
roi_image = road_lines(canny_image, mask_image)

hough_lines = cv2.HoughLinesP(roi_image, 2, np.pi / 180, 100, np.array([]), 40, 5)

# TODO: draw hough lines, average lines into one line, set distance
