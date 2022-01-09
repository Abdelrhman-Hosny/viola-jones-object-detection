from utils import compute_integral_image
from xmlparser import parse_haar_cascade_xml
import numpy as np
from skimage import feature


def getFaces(img , xml_path):
    WINDOW_SIZE = (24, 24)
    #img_gray = io.imread(img, as_gray=True)
    img_gray =  img
    img_draw = img_gray.astype(np.uint8)
    img_gray = img_gray.astype(np.uint64)
    integral_image = compute_integral_image(img_gray)
    integral_image_sqaured = compute_integral_image(np.square(img_gray))
    #curr_dir = abspath(r'.')
    #xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")
    stages, features = parse_haar_cascade_xml(xml_path)
    y_max, x_max = img_gray.shape

    # print(f'x_max : {x_max}')
    # print(f'y_max : {y_max}'

    img_canny = feature.canny(img_draw, sigma=3)
    edges = np.array(img_canny , dtype= np.uint64)
    canny_integral_image = compute_integral_image(edges)


    maxScale = int(min((y_max/24),(x_max/24)))

    faces = []

    scale = 4
    while (scale < maxScale):
        window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] * scale * scale
        canny_threshold = WINDOW_SIZE[0]*scale*scale
        step = int(scale*WINDOW_SIZE[0]*1)//7
        windowStep = int(scale * WINDOW_SIZE[0]) + 1

        for x in range(0, x_max - int(scale * WINDOW_SIZE[0]) - 1, step):
            for y in range(0, y_max - int(scale * WINDOW_SIZE[1]) - 1, step):
            
                window_canny = canny_integral_image[
                    y : y + windowStep,
                    x : x + windowStep,
                ]
                y1, x1 = 0, 0
                y2, x2 = window_canny.shape[0] - 1, window_canny.shape[1] - 1
                total_cannay = window_canny[y2, x2] + window_canny[y1, x1] - window_canny[y2, x1] - window_canny[y1, x2]


                if (total_cannay < canny_threshold):
                    continue
                
               
                window = integral_image[
                    y : y + windowStep,
                    x : x + windowStep,
                ]

                window_squared = integral_image_sqaured[
                    y : y + windowStep,
                    x : x + windowStep,
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

        scale = int(np.ceil(scale*1.25))
    
    return faces




