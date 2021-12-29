import numpy as np
from feature import Feature, Rectangle
from classifier import WeakClassifier


class Stage:
    def __init__(self, stage_threshold: int, classifier_list : list[WeakClassifier]) -> None:
        self.stage_threshold = stage_threshold
        self.classifier_list = classifier_list
        self.current_val = 0
    
    def check_stage(self, feature_list : list[Feature],  integral_image : np.ndarray) -> bool:
        for classifier in self.classifier_list:
            self.current_val += classifier.classify(feature_list,integral_image) 
            if(self.current_val < self.stage_Threshold):
                return False
        return True

 