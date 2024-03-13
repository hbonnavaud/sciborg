import numpy as np
from typing import Union, Tuple
from scipy.spatial import distance
from skimage.draw import line_aa
from skimage.transform import resize


def place_point(image: np.ndarray, position: np.ndarray, color: Union[np.ndarray, list] = np.zeros(3), radius=5):
    """
    Modify the input image
    param image: Initial image that will be modified.
    param position: Tuple[float, float], position of the point in the image as a ratio of the width and the height.
        Example: position = (.5, .5) => center
                 position = (0., 1.) => top - left corner.
    param color: Color to give to the pixels that compose the point.
    param width: Width of the circle (in pixels).
    """
    assert 0 <= position[0] <= 1 and 0 <= position[1] <= 1
    if isinstance(position, (list, tuple)):
        assert len(position) == 2
        position = np.array(position)

    # reverse Y axis so "y = 0" is the bottom of the image
    position[1] = 1 - position[1]

    center_pixel_y, center_pixel_x = (image.shape[:2] * np.flip(position)).astype(int)

    # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
    # each pixel inside this square to
    for i in range(center_pixel_x - radius, center_pixel_x + radius):
        for j in range(center_pixel_y - radius, center_pixel_y + radius):
            dist = distance.euclidean((i, j), (center_pixel_x, center_pixel_y))
            if dist < radius and 0 <= i < image.shape[1] and 0 <= j < image.shape[0]:
                image[j, i] = color
    return image


def place_line(image: np.ndarray, start: np.ndarray, stop: np.ndarray,
               color: np.ndarray = np.zeros(3)):

    """
    Modify the input image
    param image: Initial image that will be modified.
    param start: np.ndarray, position of the line's start point in the image as a ratio of the width and the
    height.
    param stop: np.ndarray, position of the line's stop point in the image as a ratio of the width and the
    height.
        Ratio example: position = (.5, .5) => center
                       position = (0., 1.) => top - left corner.

    param color: np.ndarray: Color to give to the pixels that compose the line, as a list of three integers between 0
    and 255.
    """

    if isinstance(start, (list, tuple)):
        assert len(start) == 2
        start = np.array(start)
    if isinstance(stop, (list, tuple)):
        assert len(stop) == 2
        stop = np.array(stop)

    # reverse Y axis so "y = 0" is the bottom of the image
    start[1] = 1 - start[1]
    stop[1] = 1 - stop[1]

    if isinstance(color, list):
        color = np.array(color)

    start_center_pixel_x, start_center_pixel_y = (image.shape[:2] * np.flip(start)).astype(int)
    stop_center_pixel_x, stop_center_pixel_y = (image.shape[:2] * np.flip(stop)).astype(int)
    start_center_pixel_x = min(start_center_pixel_x, image.shape[0] - 1)
    start_center_pixel_y = min(start_center_pixel_y, image.shape[1] - 1)
    stop_center_pixel_x = min(stop_center_pixel_x, image.shape[0] - 1)
    stop_center_pixel_y = min(stop_center_pixel_y, image.shape[1] - 1)

    rr, cc, val = line_aa(start_center_pixel_x, start_center_pixel_y, stop_center_pixel_x, stop_center_pixel_y)
    old = image[rr, cc]
    extended_val = np.tile(val, (3, 1)).T
    image[rr, cc] = (1 - extended_val) * old + extended_val * color


def color_from_ratio(ratio, hexadecimal=True):
    """
    Return a colour that belongs to a gradiant from red (value=0) to green (value=1).
    @param ratio: value between 0 and 1 that defines result color (0 = red, 1 = green)
    @param hexadecimal: THe colour will be return in hexadecimal if true, in a list of RGB int otherwise.
    """
    low_color = [255, 0, 0]
    high_color = [0, 255, 0]
    if hexadecimal:
        result = "#"
    else:
        result = []
    for index, (low, high) in enumerate(zip(low_color, high_color)):
        difference = high - low
        if hexadecimal:
            final_color = hex(int(low + ratio * difference))[2:]
            result += "0" + final_color if len(final_color) == 1 else final_color
        else:
            final_color = int(low + ratio * difference)
            result.append(final_color)
    return result


def resize_image(image, height=None, width=None):
    if height is None:
        assert width is not None, "At least one new dimension should be given."
        height = int(image.shape[0] * width / image.shape[1])
    else:
        width = int(image.shape[1] * height / image.shape[0])
    return resize(image / 255, (height, width), anti_aliasing=True) * 255
