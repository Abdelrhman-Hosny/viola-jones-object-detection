import numpy as np

from feature import Feature


class WeakClassifier:
    def __init__(
        self, classifier_threshold, feature_idx, left_val, right_val
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
    ) -> float:

        feature = feature_list[self.feature_idx]

        feature_val = feature.compute_feature(integral_image, window_area, scale)

        if feature_val / var < self.classifier_threshold:
            return self.left_value
        else:
            return self.right_value
