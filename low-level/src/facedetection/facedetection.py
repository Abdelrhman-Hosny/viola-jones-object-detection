from utils import compute_integral_image
from feature import Feature, Rectangle
from stage import Stage
from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import cv2

from os.path import abspath

WINDOW_SIZE = (24, 24)

image_path = abspath(r'../../../images/faces/physics.jpg')

img_gray = io.imread(image_path, as_gray=True)
img_gray = 255 * img_gray
img_gray = img_gray.astype(np.uint8)
print(np.max(img_gray))

integral_image = compute_integral_image(img_gray)

xml_path = abspath(r'../../../high-level/haar-cascades/haarcascade_frontalface_default.xml')

stages , features = parse_haar_cascade_xml(xml_path)


y_max , x_max = img_gray.shape

# print(f'x_max : {x_max}')
# print(f'y_max : {y_max}')
plt.imshow(img_gray, cmap='gray')
plt.show()

for scale in [ 1.1, 1.2]:
    print(f'Scale : {scale}')
    print('-' * 20)
    for x in range(0, x_max - int(scale * WINDOW_SIZE[0]) - 1):

        for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1):
            window = integral_image[ y:y + int(scale * WINDOW_SIZE[1]) + 1 , x:x + int(scale * WINDOW_SIZE[0]) + 1 ]
            
            face_found = True
            for c  , stage in enumerate(stages):

                if (not stage.check_stage(features, window, scale)):
                    face_found = False
                    print(f'Stage {c} failed')
                    print(f'stage threshold : {stage.stage_threshold}')
                    print(f'stage current value : {stage.stage_result}')
                    break
            
            if face_found:
                cv2.rectangle(img_gray, (x, y), (x + int(scale *  WINDOW_SIZE[0]), y + int(scale * WINDOW_SIZE[1])), (0, 255, 0), 2)

plt.imshow(img_gray, cmap='gray')
plt.show()
