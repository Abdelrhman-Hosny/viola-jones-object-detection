import xml.etree.ElementTree as ET

# important classes : Element and ElementTree
# important functions : ET.parse() (file -> ElementTree)
# ET.tostring() (Element -> string), ET.fromstring() (string -> ElementTree)

xml_path = r'./high-level/haar-cascades/haarcascade_frontalface_default.xml'
all = ET.parse(xml_path)

stages = all.find('cascade').find('stages')
print(f'# of stages : {len(stages)}')

for stage in stages:

    print(stage.find('maxWeakCount').text)
    print(stage.find('stageThreshold').text)
    classifiers = stage.find('weakClassifiers')
    for classifier in classifiers:
        print(classifier.find('internalNodes').text)
        print(classifier.find('leafValues').text)

total = 0

features = all.find('cascade').find('features')

print(f'# of features : {len(features)}')

for feature in features:

    # number of rectangles / feature (2 or 3 or 4)
    total += len(feature.find('rects'))
    print(len(feature.find('rects')))

    for rect in feature.find('rects'):
        # each rect has 5 values
        # x, y, width, height, value
        print(rect.text)
