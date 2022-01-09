from skimage.util.dtype import img_as_bool
from utils import compute_integral_image

# from feature import Feature, Rectangle
# from stage import Stage
# from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml

import numpy as np
import skimage.io as io
from skimage import feature
import matplotlib.pyplot as plt

import cv2
from timeit import default_timer as timer  

from os.path import abspath, join

from multiprocessing import Pool

CANNY_THRESHOLD_SCALE = 2.5
WINDOW_SIZE = (24, 24)

curr_dir = abspath(r'.')
#curr_dir = abspath(r'../../../.')

#image_path = join(curr_dir, r"./images/faces/physics.jpg")
image_path = join(curr_dir, r"./images/faces/man1.jpeg")
#image_path = join(curr_dir, r"./images/Neutral/image0000767.jpg")



img_gray = io.imread(image_path, as_gray=True)
img_gray = 255 * img_gray
img_gray = cv2.resize(img_gray,(640, 480))
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

img_canny = feature.canny(img_draw, sigma=3)
plt.imshow(img_canny)
plt.show()

edges = np.array(img_canny , dtype= np.uint64)
print(np.any(img_canny != False))
canny_integral_image = compute_integral_image(edges)


#, 1.1, 1.2
count = 0
start = timer()
maxScale = int(min((y_max/24),(x_max/24)))

scale = 2
while (scale < maxScale):
    print(f"Scale : {scale}")
    print("-" * 20)
    window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] * scale * scale
    faces = []
    step = int(scale*WINDOW_SIZE[0]*1)//7

    for x in range(0, x_max - int(scale * WINDOW_SIZE[0]) - 1,step):
        for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1,step):
           
            window_canny = canny_integral_image[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]
            
            y1, x1 = 0, 0
            y2, x2 = window_canny.shape[0] - 1, window_canny.shape[1] - 1

            total_cannay = window_canny[y2, x2] + window_canny[y1, x1] - window_canny[y2, x1] - window_canny[y1, x2]
            
            
            if (total_cannay < WINDOW_SIZE[0]*2.5*scale):
                continue
            
            #print("Passed")

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

                    #print(f"Stage {c} failed")
                    #print(f"stage threshold : {stage.stage_threshold}")
                    #print(f"stage current value : {stage.stage_result}")
                    break

            if face_found:
               # print("Face Found")  
               faces.append((x,y))
               cv2.rectangle(
                    img_draw,
                    (x, y),
                    (x + int(scale * WINDOW_SIZE[0]), y + int(scale * WINDOW_SIZE[1])),
                    (0, 255, 0),
                    2,
                )
            
    scale*=1.25

print(timer() - start )
plt.imshow(img_draw, cmap="gray")
plt.show()
