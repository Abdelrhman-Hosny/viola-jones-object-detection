import numpy as np

from feature import Feature, Rectangle

class WeakClassifier:
    def __init__(self, classifier_threshold, feature_idx, success_val, fail_val) -> None:
        
        self.classifier_threshold = classifier_threshold
        self.feature_idx = feature_idx  
        self.success_value = success_val
        self.fail_value = fail_val
    
    def classify(self, feature_list : list[Feature], integral_image : np.ndarray, scale: float) -> float:

        feature = feature_list[self.feature_idx]
        
        feature_val = feature.compute_feature(integral_image, scale)

        if feature_val > self.classifier_threshold:
            return self.success_value
        else:
            return self.fail_value
