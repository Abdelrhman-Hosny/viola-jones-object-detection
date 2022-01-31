import numpy as np
from math import ceil
from skimage import feature
cimport numpy as np


cpdef get_integral_image(np.ndarray[np.int64_t, ndim=2] img):
    """returns the integral image for a given img

    If you want to know more about the integral image, check out:
        https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

    paramaters:
        img: numpy array of shape (height, width)
    
    returns:
        integral_img: numpy array of shape (height, width)
    """
    return np.cumsum(np.cumsum(img, axis=0), axis=1)




cpdef get_faces(img,stages_in,features_in):
    """returns bounding boxes on the faces inside the image

    parameters:
        img: image to detect faces on
        stages_in: stages that are read from the xml file (check haar cascades xml files)
        features_in: features that are read from the xml file (check haar cascades xml files)

    returns:
        final_faces: list of bounding boxes of the faces

    """

    cdef list stages = stages_in
    cdef list features = features_in

    WINDOW_SIZE = (24, 24)
    cdef np.ndarray[np.int64_t, ndim=2] img_gray = np.array(img,dtype=np.int64)
    
    img_draw = img.astype(np.uint8)
    img_gray_2 = img_gray.copy()
   
    cdef np.ndarray[np.int64_t, ndim=2] integral_image = get_integral_image(img_gray)


    cdef np.ndarray[np.int64_t, ndim=2] integral_image_sqaured =  get_integral_image(np.square(img_gray))

    y_max_py,  x_max_py = img_gray_2.shape
    cdef int y_max = y_max_py
    cdef int x_max = x_max_py

    img_canny = feature.canny(img_draw, sigma=3)
    edges = np.array(img_canny,dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] canny_integral_image = get_integral_image(edges)
    
    # variable definitions used in looping
    cdef int maxScale = int(min((y_max/24),(x_max/24)))
    cdef int x, y, step, windowStep, lastStepX, lastStepY
    cdef int total_im, total_im_square, window_area
    faces = []
    cdef int scale = 4
    cdef np.ndarray[np.int64_t, ndim=2] window_canny, window, window_squared
    cdef float im_mean, im_var

    while (scale < maxScale):
        window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] * scale * scale
        canny_threshold = WINDOW_SIZE[0]*scale*scale
        step = int(scale*WINDOW_SIZE[0]*1)//7
        windowStep = int(scale * WINDOW_SIZE[0]) + 1
        lastStepX = x_max - int(scale * WINDOW_SIZE[0]) - 1
        lastStepY = y_max - int(scale * WINDOW_SIZE[1]) - 1
        
        rect_values = [[rect.get_bounds(scale) for rect in ft.rectangles] for ft in features]
        for x in range(0, lastStepX , step):
            for y in range(0, lastStepY, step):
            
                window_canny = canny_integral_image[
                    y : y + windowStep,
                    x : x + windowStep,
                ]
                y1, x1 = 0, 0
                y2, x2 = window_canny.shape[0] - 1, window_canny.shape[1] - 1
                
                total_canny = window_canny[y2, x2] + window_canny[y1, x1] - window_canny[y2, x1] - window_canny[y1, x2]

                if (total_canny < canny_threshold):
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
                im_var = total_im_square / window_area - im_mean * im_mean
                im_var = np.sqrt(im_var)

                if im_var < 1:
                    im_var = 1

                face_found = True
                for stage in stages:
                    if not stage.check_stage(features, window, window_area, im_var, scale, rect_values):
                        face_found = False
                        break

                if face_found:
                   faces.append((x,y,x + WINDOW_SIZE[0]* scale, y + WINDOW_SIZE[1]* scale))

        scale = int(ceil(scale*1.25))
    

    # non max suppression to remove overlapping faces
    final_faces = []
    for i in range(len(faces)):
        main_rect = faces[i]
        for j in range(i + 1, len(faces)):
            sub_rect = faces[j]
            found = False
            # 0 is left
            # 1 is top
            # 2 is right
            # 3 is bottom

            if(
                # x overlap
                (
                    sub_rect[0] < main_rect[0] < sub_rect[2] or
                    sub_rect[0] < main_rect[2] < sub_rect[2]
                )
                and # y overlap
                (
                    sub_rect[1] < main_rect[1] < sub_rect[3] or
                    sub_rect[1] < main_rect[3] < sub_rect[3]
                )
            ):
               
               # if overlapping remove the one with smaller scale
                if(faces[i][2] < faces[j][2]):
                   found = True
                   final_faces.append(faces[j])
                else:
                   found = True
                   final_faces.append(faces[i])

        if not found:
            final_faces.append(faces[i])

    return final_faces




