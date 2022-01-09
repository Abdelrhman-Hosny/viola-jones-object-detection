import numpy as np


class Rectangle:
    def __init__(self, x: int, y: int, w: int, h: int, val: int) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        self.val = val

    def get_bounds(self, scale) -> tuple[int, int, int, int]:
        """
        (x1,y1) is the top left corner of the rectangle
        (x2,y2) is the bottom right corner of the rectangle
        Returns the bounds of the rectangle.
        """
        return (
            int(self.x1 * scale),
            int(self.y1 * scale),
            int(self.x2 * scale),
            int(self.y2 * scale),
        )


class Feature:
    def __init__(self, rect_array: list[Rectangle]) -> None:
        """
        Initializes a feature with a list of rectangles.
        There are 4 types of features:
            1. 2 Rectangles stacked horizontally
            2. 2 Rectangles stacked vertically
            2. 3 Rectangles stacked horizontally
            3. 4 Rectangles stacked each two stacked next to each other.
        :param rect_array: The list of rectangles that make up the feature.
        """
        self.rectangles = rect_array

    # (features, window, window_squared, window_area, var, scale):
    def compute_feature(
        self, integral_image: np.ndarray, window_area: float, scale: float = 1,
    ) -> float:
        """
        Computes the feature of the feature.
        :param image: The integral image to compute the feature of.
        :return: The sum of the rectangles in the feature
        """

        feature_sum = 0

        for rect in self.rectangles:
            x1, y1, x2, y2 = rect.get_bounds(scale)

            feature_sum += (
                integral_image[y2, x2]  # bottom right
                - integral_image[y1, x2]  # top right
                - integral_image[y2, x1]  # bottom left
                + integral_image[y1, x1]  # top left
            ) * rect.val

        return feature_sum 
