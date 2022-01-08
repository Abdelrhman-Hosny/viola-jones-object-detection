import functools
from utils import compute_integral_image
# from feature import Feature, Rectangle
from stage import Stage
from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml,parse_haar_cascade_xml2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from utils import compute_integral_image
from xmlparser import parse_haar_cascade_xml,parse_haar_cascade_xml2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import feature
import numba as nb
import cv2
from os.path import abspath, join 
from multiprocessing import Pool
from itertools import repeat
from timeit import default_timer as timer  



def DetectFaceMP(WINDOW_SIZE , scale , stages , features ,window, window_squard, Canny_window, X_Y):

    window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1]* scale * scale
    y1, x1 = 0, 0
    y2, x2 = window.shape[0] - 1, window.shape[1] - 1

    total_cannay = Canny_window[y2, x2] + Canny_window[y1, x1] - Canny_window[y2, x1] - Canny_window[y1, x2]

    if (total_cannay < WINDOW_SIZE[0]*2.5*scale):
        return
    
    total_im = window[y2, x2] + window[y1, x1] - window[y2, x1] - window[y1, x2]

    total_im_square = (
        window_squard[y2, x2]
        + window_squard[y1, x1]
        - window_squard[y2, x1]
        - window_squard[y1, x2]
    )

    im_mean = total_im / window_area

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
        return X_Y
    


if __name__ == '__main__':
    start = timer()
    print("start")
    scale = 1
    CANNY_THRESHOLD_SCALE = 2.5
    WINDOW_SIZE = (24, 24)
    curr_dir = abspath(r'.')


    #curr_dir = abspath(r'../../../.')

    image_path = join(curr_dir, r"./images/faces/physics.jpg")
    img_gray = io.imread(image_path, as_gray=True)
    img_gray = 255 * img_gray
    img_gray = cv2.resize(img_gray,(640, 480))
    img_draw = img_gray.astype(np.uint8)
    img_gray = img_gray.astype(np.uint64)

    integral_image = compute_integral_image(img_gray)
    integral_image_sqaured = compute_integral_image(np.square(img_gray))

    xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")

    img_canny = feature.canny(img_draw, sigma=3)

    edges = np.array(img_canny , dtype= np.uint64)

    canny_integral_image = compute_integral_image(edges)

    #stages contains vector of classifiers
    stages, features = parse_haar_cascade_xml(xml_path)
    y_max, x_max = img_gray.shape

    
    #for scale in [1]:
    AllWindows = []
    AllWindows_Squard = []
    AllCannyWindows = []
    AllX_Y = []
    for x in range(0, x_max - int(scale * WINDOW_SIZE[0])  - 1):
        for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1):
            window = integral_image[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]
            window_squared = integral_image_sqaured[
                y : y + int(scale * WINDOW_SIZE[1]) + 1,
                x : x + int(scale * WINDOW_SIZE[0]) + 1,
            ]
            window_canny = canny_integral_image[
                    y : y + int(scale * WINDOW_SIZE[1]) + 1,
                    x : x + int(scale * WINDOW_SIZE[0]) + 1,
                ]
            AllWindows.append(window)
            AllWindows_Squard.append(window_squared)
            AllCannyWindows.append(window_canny)
            X_Y =[x,y]
            AllX_Y.append(X_Y)
    AllWindows_np = np.array(AllWindows, dtype = np.uint64)
    AllWindow_Squared_np = np.array(AllWindows_Squard, dtype = np.uint64)
    AllCannyWindow_np = np.array(AllCannyWindows, dtype = np.uint64)
    print(AllWindows_np.shape[0])
    #partial_DetectFaceMP = functools.partial(DetectFaceMP,(WINDOW_SIZE),(scale),(stages), (features))
    pool = Pool()
    Allfaces = pool.starmap(DetectFaceMP, zip(repeat(WINDOW_SIZE),repeat(scale),repeat(stages), repeat(features) , AllWindows_np,AllWindow_Squared_np,AllCannyWindow_np,AllX_Y))
    #pool.close()
    #pool.join()
    Not_none_values = filter(None.__ne__, Allfaces)
    faces = list(Not_none_values)
    for face in faces:
        cv2.rectangle(
             img_draw,
             (face[0], face[1]),
             (face[0] + int(scale * WINDOW_SIZE[0]), face[1] + int(scale * WINDOW_SIZE[1])),
             (0, 255, 0),
             2,
         )
    plt.imshow(img_draw, cmap="gray")
    plt.show()

