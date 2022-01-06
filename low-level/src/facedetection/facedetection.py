from utils import compute_integral_image

# from feature import Feature, Rectangle
# from stage import Stage
# from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import cv2

from os.path import abspath, join

WINDOW_SIZE = (20, 20)

curr_dir = abspath(r'.')
# curr_dir = abspath(r'../../../.')

image_path = join(curr_dir, r"./images/faces/physics.jpg")

img_gray = io.imread(image_path, as_gray=True)
img_gray = 255 * img_gray
img_draw = img_gray.astype(np.uint8)
img_gray = img_gray.astype(np.uint64)
print(np.max(img_gray))

integral_image = compute_integral_image(img_gray)

integral_image_sqaured = compute_integral_image(np.square(img_gray))

xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")

stages, features = parse_haar_cascade_xml(xml_path)


y_max, x_max = img_gray.shape

# print(f'x_max : {x_max}')
# print(f'y_max : {y_max}')
plt.imshow(img_draw, cmap="gray")
plt.show()

for scale in [1, 1.1, 1.2]:
    print(f"Scale : {scale}")
    print("-" * 20)
    window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] * scale * scale

    for x in range(0, x_max - int(scale * WINDOW_SIZE[0]) - 1):
        for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1):

            window = integral_image[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]

            window_squared = integral_image_sqaured[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]

            y1, x1 = 0, 0
            y2, x2 = window.shape[0] - 1, window.shape[1] - 1

            total_im = window[y2, x2] + window[y1, x1] - window[y2, x1] - window[y1, x2]

            total_im_square = (
                window_squared[y2, x2]
                + window_squared[y1, x1]
                - window_squared[y2, x1]
                - window_squared[y1, x2]
            )

            im_mean = total_im / window_area
            # print(total_im_square/window_area - im_mean * im_mean)
            im_var = total_im_square / window_area - im_mean * im_mean
            im_var = np.sqrt(im_var)

            if im_var < 1:
                im_var = 1

            face_found = True
            for c, stage in enumerate(stages):

                if not stage.check_stage(features, window, window_area, im_var, scale):
                    face_found = False
                    if c < 3:
                        break

                    print(f"Stage {c} failed")
                    print(f"stage threshold : {stage.stage_threshold}")
                    print(f"stage current value : {stage.stage_result}")
                    break

            if face_found:
                print("Face Found")
                cv2.rectangle(
                    img_draw,
                    (x, y),
                    (x + int(scale * WINDOW_SIZE[0]), y + int(scale * WINDOW_SIZE[1])),
                    (0, 255, 0),
                    2,
                )

plt.imshow(img_draw, cmap="gray")
plt.show()
