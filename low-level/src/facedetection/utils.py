import numpy as np


def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """
    Computes the integral image of an image.
    :param image: The grayscale image of shape (w,h) to compute the integral
                  image of.
    :return: The integral image.
    """

    return np.cumsum(np.cumsum(image, axis=0), axis=1)
