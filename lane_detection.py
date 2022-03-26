"""Detect straight lanes of a road using Hough transforms."""

import cv2
import numpy as np
from typing import Tuple


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
    triangle = np.array([[(230, shape[0]), (1100, shape[0]), (550, 200)]])

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


def average_line(image: np.ndarray, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get average coordinates of line out of set of lines.

    Args:
        image: Image that contains the edges/lines.
        lines: Parameters (slope and intercept) of lines found by Hough algorithm.

    Returns:
        Tuple containing the coordinates of the left lane and right lane.
    """
    left_lane = []
    right_lane = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_lane.append((slope, intercept))
        else:
            right_lane.append((slope, intercept))

    avg_left = np.mean(left_lane, axis=0)
    avg_right = np.mean(right_lane, axis=0)
    left_coordinates = _coordinates_from_slope(image, avg_left)
    right_coordinates = _coordinates_from_slope(image, avg_right)

    return left_coordinates, right_coordinates


def _coordinates_from_slope(
    image: np.ndarray, line_parameters: Tuple[float, float]
) -> np.ndarray:
    """Auxiliary function to obtain coordinates in image given slope and incercept.

    Args:
        image: Image that contains the edges/lines.
        line_parameters: Tuple containing average slope and intercept.

    Returns:
        X and y coordinates in image.
    """
    slope, intercept = line_parameters

    start_height = image.shape[0]
    end_height = int(start_height * (3 / 5))

    start_width = int((start_height - intercept) / slope)
    end_width = int((end_height - intercept) / slope)

    return np.array([start_width, start_height, end_width, end_height])


def draw_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Draw lines on image.

    Args:
        image: Image that contains the edges/lines.
        lines: Coordinates of lines to draw.

    Returns:
        Mask with lines drawn on it.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def fill_road(image: np.ndarray, lines: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Fill the area between two lines.

    Args:
        image: Image that contains the edges/lines.
        lines: Coordinates of left and right lane.

    Returns:
        Mask with filled area.
    """

    x1, y1, x2, y2 = lines[0]
    x4, y4, x3, y3 = lines[1]
    mask = np.zeros_like(image)
    cv2.fillPoly(
        mask, np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]), (0, 255, 0)
    )
    return mask


def detect_lanes_in_image(image_path: str) -> None:
    """Find lanes in an image.

    Args:
        image_path: Path of image.
    """
    # Load image
    image = cv2.imread(image_path)
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
    average_lines = average_line(color_image, hough_lines)
    line_img = draw_lines(color_image, average_lines)

    combined_image = cv2.addWeighted(color_image, 0.8, line_img, 1, 1)

    mask = fill_road(combined_image, average_lines)

    result = cv2.addWeighted(combined_image, 0.9, mask, 0.4, 1)

    cv2.imshow("Detected lanes", result)
    if cv2.waitKey(0) == ord("q"):
        return


if __name__ == "__main__":
    detect_lanes_in_image("test_image.jpg")
