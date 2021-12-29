import xml.etree.ElementTree as ET

# important classes : Element and ElementTree
# important functions : ET.parse() (file -> ElementTree)
# ET.tostring() (Element -> string), ET.fromstring() (string -> ElementTree)

xml_path = r'./high-level/haar-cascades/haarcascade_frontalface_default.xml'
all = ET.parse(xml_path)

stages = all.find('cascade').find('stages')
print(f'# of stages : {len(stages)}')

num_classifiers = 0

for stage in stages:

    # print(stage.find('maxWeakCount').text)
    # print(stage.find('stageThreshold').text)
    # stage_threshold = float(stage.find('stageThreshold').text)
    classifiers = stage.find('weakClassifiers')
    
    num_classifiers += len(classifiers)
    
    for classifier in classifiers:

        # print(classifier.find('internalNodes').text)
        internal_nodes = (classifier.find('internalNodes').text).split()
        leaf_values = (classifier.find('leafValues').text).split()
        val_to_add_success , val_to_add_fail = leaf_values
        _ , _ , feature_idx , node_threshold = internal_nodes

        # what will we do with these?
        # if (feature[feature_idx] > node_threshold):
        #     val_to_add = val_to_add_success
        # else:
        #     val_to_add = val_to_add_fail
        # stage_accumulator += val_to_add
        # and we repeat that for each node, until we finish the stage
        # if we succeed we pass to the new stage
        # if we fail, we exit and move on to next window.

print(f'# of classifiers : {num_classifiers}')

features = all.find('cascade').find('features')

print(f'# of features : {len(features)}')

for feature in features:

    # number of rectangles / feature (2 or 3 or 4)
    print(len(feature.find('rects')))

    for rect in feature.find('rects'):
        # each rect has 5 values
        # x, y, width, height, value
        print(rect.text)
