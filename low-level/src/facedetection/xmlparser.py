from typing import List
import xml.etree.ElementTree as ET
from classifier import WeakClassifier
from feature import Feature, Rectangle
from stage import Stage

def parse_haar_cascade_xml(xml_path : str ) -> tuple[list[Stage], list[Feature]]:
    """Reads xml file and returns a list of stages and a list of features."""
    
    all = ET.parse(xml_path)

    stages = all.find('cascade').find('stages')

    my_stages = [] 

    for stage in stages:
        
        classifiers = stage.find('weakClassifiers')
        my_classifiers = []
        for classifier in classifiers:

            # print(classifier.find('internalNodes').text)
            internal_nodes = (classifier.find('internalNodes').text).split()
            leaf_values = (classifier.find('leafValues').text).split()
            
            # print(f'leafValues : {leafValues}')
            _ , _ , feature_idx , node_threshold = internal_nodes
            success_val , fail_val = leaf_values

            my_classifiers.append(WeakClassifier(node_threshold, feature_idx, success_val, fail_val))    
        
        # print(stage.find('stageThreshold').text)
        stage_threshold = float(stage.find('stageThreshold').text)

        my_stages.append(Stage(stage_threshold, classifiers))

    features = all.find('cascade').find('features')

    my_features = []

    for feature in features:

        my_rectangles = []

        for rect in feature.find('rects'):
            # each rect has 5 values
            # x, y, width, height, value

            my_rect = rect.text.split()
            my_rect = [float(x) for x in my_rect]
            my_rectangles.append(Rectangle(*my_rect))

        my_features.append(Feature(my_rectangles))

    return my_stages, my_features