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
from itertools import repeat
from stage import Stage
from classifier import WeakClassifier


def DetectFaces(scale ,WINDOW_SIZE ,canny_integral_image , integral_image , integral_image_sqaured , stages ,features ,x_max,y_max):

    window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] * scale * scale
    faces = []
    step = int(scale*WINDOW_SIZE[0]*1)//7

    for x in range(0, x_max - int(scale * WINDOW_SIZE[0]) - 1,step):
        for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1,step):
            #Dup = False
            #for face in faces:
            #    if (x < face[0] + WINDOW_SIZE[0] * scale /2) and (y < face[1]  + WINDOW_SIZE[0] * scale  /2):
            #        Dup = True
            #        break
            #if Dup:
            #    continue

            window_canny = canny_integral_image[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]
            
            y1, x1 = 0, 0
            y2, x2 = window_canny.shape[0] - 1, window_canny.shape[1] - 1

            total_cannay = window_canny[y2, x2] + window_canny[y1, x1] - window_canny[y2, x1] - window_canny[y1, x2]
            
            
            if (total_cannay < WINDOW_SIZE[0]*scale*scale):
                continue
            
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
            for stage in stages:
                if not stage.check_stage(features, window, window_area, im_var, scale):
                    face_found = False
                    break

            if face_found:
               faces.append((x,y,scale))
    
    return faces
            
            


if __name__ == '__main__':
    WINDOW_SIZE = (24, 24)

    curr_dir = abspath(r'.')
    #curr_dir = abspath(r'../../../.')

    image_path = join(curr_dir, r"./images/faces/Ali.jpg")
    #image_path = join(curr_dir, r"./images/faces/man1.jpeg")


    img_gray = io.imread(image_path, as_gray=True)
    img_gray = 255 * img_gray
    img_gray = cv2.resize(img_gray,(640, 480))
    img_draw = img_gray.astype(np.uint8)
    img_gray = img_gray.astype(np.uint64)

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
    edges = np.array(img_canny , dtype= np.uint64)
    canny_integral_image = compute_integral_image(edges)


    #, 1.1, 1.2
    start = timer()
    maxScale = int(min((y_max/24),(x_max/24)))

    scale = 2
    AllScales = []
    while (scale < maxScale):
        AllScales.append(scale)
        scale = int(np.ceil(scale*1.25))


    start = timer()
    pool = Pool()
    Allfaces = pool.starmap(DetectFaces,zip(AllScales,repeat(WINDOW_SIZE), repeat(canny_integral_image) , repeat(integral_image) ,repeat(integral_image_sqaured) , repeat(stages) , repeat(features) ,repeat(x_max) ,repeat(y_max) ))
    print(timer() - start )
    faces = list2 = [x for x in Allfaces if x != []]
    for face in faces:
            cv2.rectangle(
             img_draw,
             (face[0][0], face[0][1]),
             (face[0][0] + int(face[0][2] * WINDOW_SIZE[0]), face[0][1] + int(face[0][2] * WINDOW_SIZE[1])),
             (0, 255, 0),
             2,
         )

    plt.imshow(img_draw, cmap="gray")
    plt.show()
