from utils import compute_integral_image

# from feature import Feature, Rectangle
# from stage import Stage
# from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml,parse_haar_cascade_xml2

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import feature


import numba as nb

import cv2
import math

from os.path import abspath, join

from numba import cuda 
from timeit import default_timer as timer  

CANNY_THRESHOLD_SCALE = 2.5

WINDOW_SIZE = (24, 24)

@cuda.jit
def CalculateFeatures(ClassifierList,ClassifierList_count,im_var,window_area,window,classifiers_result,feature_list):
      for index , classifiers in enumerate(ClassifierList):
        if(index > ClassifierList_count-1):
            break
        feature_sum = 0
        for feature in feature_list[index]:
             # each rect has 5 values
             # x, y, width, height, value
            x1, y1, x2, y2 = feature[0],feature[1],feature[2],feature[3]
            feature_sum += (
                window[y2, x2] # bottom right
                - window[y1, x2]  # top right
                - window[y2, x1]  # bottom left
                + window[y1, x1]  # top left
            ) * feature[4]
        feature_sum = feature_sum / window_area
        if(feature_sum / im_var < float(classifiers[0])):
            classifiers_result[index] += classifiers[2]
        else:
            classifiers_result[index] += classifiers[3]


#@cuda.jit    
def DetectFace(stages_np,features_np,my_stages_threshold_np,WINDOW_SIZE,Stage_classifier_Count,AllWindows_np,AllWindow_Squared_np,face_indexes,AllCannyWindow_np):
    #for scale in [1, 1.1, 1.2]:   
    #print(f"Scale : {scale}")
    #print("-" * 20)* scale * scale

    window_area = WINDOW_SIZE[0] * WINDOW_SIZE[1] 
    index = 0
    for window_index, window in enumerate(AllWindows_np):
        # for y in range(0, y_max - int(WINDOW_SIZE[1]) - 1):
        #     window = integral_image[
        #         y : y + int(scale * WINDOW_SIZE[1]) + 1,
        #         x : x + int(scale * WINDOW_SIZE[0]) + 1,
        #     ]
        #     window_squared = integral_image_sqaured[
        #         y : y + int(scale * WINDOW_SIZE[1]) + 1,
        #         x : x + int(scale * WINDOW_SIZE[0]) + 1,
        #     ]
            y1, x1 = 0, 0
            y2, x2 = window.shape[0] - 1, window.shape[1] - 1
            total_cannay = AllCannyWindow_np[window_index][y2, x2] + AllCannyWindow_np[window_index][y1, x1] - AllCannyWindow_np[window_index][y2, x1] - AllCannyWindow_np[window_index][y1, x2]
            if (total_cannay < WINDOW_SIZE[0]*2.5*scale):
                continue
            
            total_im = window[y2, x2] + window[y1, x1] - window[y2, x1] - window[y1, x2]
            total_im_square = (
                AllWindow_Squared_np[window_index][y2, x2]
                + AllWindow_Squared_np[window_index][y1, x1]
                - AllWindow_Squared_np[window_index][y2, x1]
                - AllWindow_Squared_np[window_index][y1, x2]
            )
            im_mean = total_im / window_area
            # print(total_im_square/window_area - im_mean * im_mean)
            im_var = total_im_square / window_area - im_mean * im_mean
            im_var = math.sqrt(im_var)
            if im_var < 1:
                im_var = 1
            face_found = True
            for c, stage in enumerate(stages_np):
                ClassifierList = stage
                ClassifierList_count= Stage_classifier_Count[c]
                featuresList = features_np[int(ClassifierList[0][1]):int(ClassifierList[ClassifierList_count-1][1])+1]
                classifiers_result = 0
                classifiers_result =np.zeros(211)
                device = cuda.get_current_device()
                threadsperblock = device.WARP_SIZE
                blockperthread = int(np.ceil(float(211)/threadsperblock))
                GPU_featureList = cuda.to_device(featuresList)
                GPU_classifiers_result = cuda.to_device(classifiers_result)
                GPU_ClassifierList = cuda.to_device(ClassifierList)
                windows_np = np.array(window)
                GPU_Window = cuda.to_device(windows_np)
                CalculateFeatures[blockperthread, threadsperblock](GPU_ClassifierList,ClassifierList_count,im_var,window_area,GPU_Window,GPU_classifiers_result,GPU_featureList)
                classifiers_result = GPU_classifiers_result.copy_to_host()
                #for index , classifiers in enumerate(ClassifierList):
                #    if(index > ClassifierList_count-1):
                #        break
                #    feature_sum = 0
                #    for feature in features_np[int(classifiers[1])]:
                #         # each rect has 5 values
                #         # x, y, width, height, value
                #        x1, y1, x2, y2 = feature[0],feature[1],feature[2],feature[3]
                #        feature_sum += (
                #            window[y2, x2] # bottom right
                #            - window[y1, x2]  # top right
                #            - window[y2, x1]  # bottom left
                #            + window[y1, x1]  # top left
                #        ) * feature[4]
                #    feature_sum = feature_sum / window_area
                #    if(feature_sum / im_var < float(classifiers[0])):
                #        classifiers_result += classifiers[2]
                #    else:
                #        classifiers_result += classifiers[3]

                #current_stage_result = classifiers_result
                current_stage_result = sum(classifiers_result)
                if(current_stage_result < my_stages_threshold_np[c]):
                    face_found = False
                    #if c < 3:
                    #    break
                    #print(f"Stage {c} failed")
                    #print(f"stage threshold : {c}")
                    #print(f"stage current value : {current_stage_result}")
                    #break
            if face_found:
                print("face Found")
                #face_indexes[index] = -1
                #print("Face Found")
                #cv2.rectangle(
                #     img_draw,
                #     (x, y),
                #     (x + int(scale * WINDOW_SIZE[0]), y + int(scale * WINDOW_SIZE[1])),
                #     (0, 255, 0),
                #     2,
                # )
                #img_draw[x][y] = 0
            index+=1

curr_dir = abspath(r'.')
#curr_dir = abspath(r'../../../.')

image_path = join(curr_dir, r"./images/faces/physics.jpg")

img_gray = io.imread(image_path, as_gray=True)
img_gray = 255 * img_gray
img_draw = img_gray.astype(np.uint8)
img_gray = img_gray.astype(np.uint64)
print(np.max(img_gray))

integral_image = compute_integral_image(img_gray)

integral_image_sqaured = compute_integral_image(np.square(img_gray))

xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")

img_canny = feature.canny(img_draw, sigma=3)
edges = np.array(img_canny , dtype= np.uint64)
canny_integral_image = compute_integral_image(edges)


#stages contains vector of classifiers
stages, features,my_stages_threshold,stage_classifiers_count = parse_haar_cascade_xml2(xml_path)

my_stages_threshold_np = np.array(my_stages_threshold , dtype=np.float32)
print(my_stages_threshold_np)
stages_np = np.array(stages, dtype= np.float32)
features_np = np.array(features, dtype=np.int32)
stage_classifiers_count_np = np.array(stage_classifiers_count,dtype=np.int32)
y_max, x_max = img_gray.shape

plt.imshow(img_draw, cmap="gray")
plt.show()



start = timer()

scale = 1

print("start")



AllWindows = []
AllWindows_Squard = []
AllCannyWindows = []
for x in range(0, x_max - int(WINDOW_SIZE[0]) - 1):
    for y in range(0, y_max - int(WINDOW_SIZE[1]) - 1):
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



print("finish")

AllWindows_np = np.array(AllWindows, dtype = np.int32)
AllWindow_Squared_np = np.array(AllWindows_Squard, dtype = np.int32)
AllCannyWindow_np = np.array(AllCannyWindows, dtype = np.int32)


print(AllWindows_np.shape[0])

#GPU_AllWindows_np = cuda.to_device(AllWindows_np)
#GPU_AllWindow_Squard = cuda.to_device(AllWindow_Squared_np)
#device = cuda.get_current_device()
#threadsperblock = device.WARP_SIZE
#blockperthread = int(np.ceil(float(AllWindows_np.shape[0])/threadsperblock))
#GPU_stages = cuda.to_device(stages_np)
#GPU_features = cuda.to_device(features_np)
#GPU_my_stages_threshold_np = cuda.to_device(my_stages_threshold_np)
#GPU_stage_classifiers_count = cuda.to_device(stage_classifiers_count_np)
#GPU_WINDOW_SIZE =  cuda.to_device(WINDOW_SIZE)
#
faceIndex = np.zeros(int(np.ceil(float(y_max*x_max)/(WINDOW_SIZE[0]*WINDOW_SIZE[1]))))
#
#GPU_face_indexes = cuda.to_device(faceIndex)

#DetectFace[blockperthread, threadsperblock](GPU_stages,GPU_features,GPU_my_stages_threshold_np,GPU_WINDOW_SIZE,GPU_stage_classifiers_count,GPU_AllWindows_np,GPU_AllWindow_Squard,GPU_face_indexes)


DetectFace(stages_np,features_np,my_stages_threshold_np,WINDOW_SIZE,stage_classifiers_count,AllWindows,AllWindows_Squard,faceIndex,AllCannyWindow_np)

#faceIndex = GPU_face_indexes.copy_to_host()

#print(faceIndex)


plt.imshow(img_draw, cmap="gray")
plt.show()
