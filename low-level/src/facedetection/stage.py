import numpy as np
from feature import Feature, Rectangle
from classifier import WeakClassifier


class Stage:
    def __init__(self, stage_Threshold: int, classfier_list : list[WeakClassifier]) -> None:
        self.stage_Threshold = stage_Threshold
        self.classfier_list = classfier_list
        self.current_val = 0
    
    def check_stage(self, feature_list : list[Feature],  integral_image : np.ndarray) -> bool:
        for classifer in self.classfier_list:
            self.current_val += classifer.classify(feature_list,integral_image) 
            if(self.current_val < self.stage_Threshold):
                return False
        return True

 