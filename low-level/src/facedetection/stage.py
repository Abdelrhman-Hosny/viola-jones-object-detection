import numpy as np
from feature import Feature
from classifier import WeakClassifier


class Stage:
    def __init__(
        self, stage_threshold: int, classifier_list: list[WeakClassifier]
    ) -> None:
        self.stage_threshold = stage_threshold
        self.classifier_list = classifier_list

    def check_stage(
        self,
        feature_list: list[Feature],
        integral_image: np.ndarray,
        window_area: float,
        var: float,
        scale: float,
        rect_values: list[int],
    ) -> bool:

        stage_result = [
            clf.classify(
                        feature_list,
                        integral_image,
                        window_area,
                        var,
                        scale,
                        rect_values,
            )

            for clf in self.classifier_list
        ]

        self.stage_result = sum(stage_result)
        return self.stage_result > self.stage_threshold
