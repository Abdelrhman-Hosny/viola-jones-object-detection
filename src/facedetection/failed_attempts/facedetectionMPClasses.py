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



def DetectFaceMP(scale , stages , features ,window, X_Y , window_area,im_var):
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
  
    CANNY_THRESHOLD_SCALE = 2.5
    WINDOW_SIZE = (24, 24)
    curr_dir = abspath(r'.')


    #curr_dir = abspath(r'../../../.')

    #image_path = join(curr_dir, r"./images/faces/physics.jpg")
    image_path = join(curr_dir, r"./images/faces/man1.jpeg")

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
    maxScale = int(min((y_max/24),(x_max/24)))
    scale = 2
    start = timer()
    while (scale < maxScale):
        AllWindows = []
        AllX_Y = []
        All_IM = []
        window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1]* scale * scale
        step = int(scale*WINDOW_SIZE[0]*1)//7

        for x in range(0, x_max - int(scale * WINDOW_SIZE[0])  - 1,step):
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


                window = integral_image[
                    y : y + int(scale * WINDOW_SIZE[1]) + 1,
                    x : x + int(scale * WINDOW_SIZE[0]) + 1,
                ]
                window_squared = integral_image_sqaured[
                    y : y + int(scale * WINDOW_SIZE[1]) + 1,
                    x : x + int(scale * WINDOW_SIZE[0]) + 1,
                ]
                total_im = window[y2, x2] + window[y1, x1] - window[y2, x1] - window[y1, x2]
                total_im_square = (
                    window_squared[y2, x2]
                    + window_squared[y1, x1]
                    - window_squared[y2, x1]
                    - window_squared[y1, x2]
                )
                im_mean = total_im / window_area
                im_var = total_im_square / window_area - im_mean * im_mean

                im_var = np.sqrt(im_var)

                if im_var < 1:
                    im_var = 1

                AllWindows.append(window)
                All_IM.append(im_var)
                X_Y =[x,y]
                AllX_Y.append(X_Y)


            #print(len(AllWindows))
            #print(timer() - start)
            pool = Pool()
            Allfaces = pool.starmap(DetectFaceMP,zip(repeat(scale),repeat(stages), repeat(features) , AllWindows,AllX_Y, repeat(window_area) , All_IM))
      

           
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
            scale *= 1.25


    plt.imshow(img_draw, cmap="gray")
    plt.show()
    print(timer() - start)

     ####NP array Slower is multiprocessoring than the list ####
    #AllWindows_np = np.array(AllWindows, dtype = np.uint64)
    #AllWindow_Squared_np = np.array(AllWindows_Squard, dtype = np.uint64)
    #AllCannyWindow_np = np.array(AllCannyWindows, dtype = np.uint64)
    #print(AllWindows_np.shape[0])
    ##partial_DetectFaceMP = functools.partial(DetectFaceMP,(WINDOW_SIZE),(scale),(stages), (features))
    #pool = Pool()
    #start = timer()
    #Allfaces = pool.starmap(DetectFaceMP, zip(repeat(WINDOW_SIZE),repeat(scale),repeat(stages), repeat(features) , AllWindows_np,AllWindow_Squared_np,AllX_Y))
    #print("NP array")
    #print(timer() - start)
