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
        self, feature_list: list[Feature],
        integral_image: np.ndarray,
        scale: float
    ) -> bool:

        stage_result = [
            clf.classify(feature_list, integral_image, scale)
            for clf in self.classifier_list
        ]

        self.stage_result = sum(stage_result)
        return self.stage_result > self.stage_threshold
