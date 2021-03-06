import numpy as np

from facedetection.blocks.feature import Feature


class WeakClassifier:
    def __init__(
        self,
        classifier_threshold: float,
        feature_idx: int,
        left_val: float,
        right_val: float,
    ) -> None:

        self.classifier_threshold = classifier_threshold
        self.feature_idx = feature_idx
        self.left_value = left_val
        self.right_value = right_val

    # (features, window, window_squared, window_area, var, scale):
    def classify(
        self,
        feature_list: list[Feature],
        integral_image: np.ndarray,
        window_area: float,
        var: float,
        scale: float,
        rect_values: list[int],
    ) -> float:

        feature_idx = self.feature_idx
        feature = feature_list[feature_idx]

        feature_val = feature.compute_feature(
                                            integral_image,
                                            window_area,
                                            scale,
                                            rect_values[feature_idx],
                                            )

        if feature_val < self.classifier_threshold * var:
            return self.left_value
        else:
            return self.right_value
