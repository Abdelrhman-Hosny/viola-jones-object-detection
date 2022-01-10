from utils import compute_integral_image
import numpy as np
from skimage import feature
cimport numpy as np
from timeit import default_timer as timer  



cpdef getIntegralImageCY(np.ndarray[np.int64_t, ndim=2] img):
      return np.cumsum(np.cumsum(img, axis=0), axis=1)




cpdef getFaces(img,stages_in,features_in):
    cdef list stages
    cdef list features
    stages = stages_in
    features = features_in

    WINDOW_SIZE = (24, 24)
    #img_gray = io.imread(img, as_gray=True)
    cdef np.ndarray[np.int64_t, ndim=2] img_gray
    img_gray = np.array(img,dtype=np.int64)
    img_draw = img.astype(np.uint8)
    img_gray_2 = img_gray.astype(np.uint64)
   
    cdef np.ndarray[np.int64_t, ndim=2] integral_image
    integral_image =getIntegralImageCY(img_gray)


    cdef np.ndarray[np.int64_t, ndim=2] integral_image_sqaured

    integral_image_sqaured = np.array(getIntegralImageCY(np.square(img_gray)),dtype=np.int64)

    #curr_dir = abspath(r'.')
    #xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")


    y_max_py,  x_max_py = img_gray_2.shape
    cdef int y_max = y_max_py
    cdef int x_max = x_max_py
    # print(f'x_max : {x_max}')
    # print(f'y_max : {y_max}'

    img_canny = feature.canny(img_draw, sigma=3)
    edges = np.array(img_canny,dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] canny_integral_image
    canny_integral_image = getIntegralImageCY(edges)

    cdef int maxScale = int(min((y_max/24),(x_max/24)))
    cdef int x
    cdef int y
    cdef int step 
    cdef int windowStep 
    cdef int lastStepX 
    cdef int lastStepY 
    faces = []

    cdef int scale = 4
    cdef np.ndarray[np.int64_t, ndim=2] window_canny
    cdef np.ndarray[np.int64_t, ndim=2] window
    cdef np.ndarray[np.int64_t, ndim=2] window_squared
    cdef int total_im
    cdef int total_im_square
    cdef float im_mean
    cdef float im_var
    cdef int window_area

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
                    if not stage.check_stage(features, window, window_area, im_var, scale, rect_values):
                        face_found = False
                        break

                if face_found:
                   faces.append((x,y,scale))

        scale = int(np.ceil(scale*1.25))
    

    finalFaces = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
           found = False
           if(abs(faces[i][0] - faces[j][0] < 24 * faces[i][2]) and abs(faces[i][1] - faces[j][1] < 24 * faces[i][2])):
               #overlapping remove the one with smaller scale
               if(faces[i][2] < faces[j][2]):
                   found = True
                   finalFaces.append(faces[j])
               else:
                   found = True
                   finalFaces.append(faces[i])
        if not found:
            finalFaces.append(faces[i])

    return finalFaces




