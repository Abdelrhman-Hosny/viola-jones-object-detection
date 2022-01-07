from utils import compute_integral_image

# from feature import Feature, Rectangle
# from stage import Stage
# from classifier import WeakClassifier
from xmlparser import parse_haar_cascade_xml,parse_haar_cascade_xml2

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import numba as nb

import cv2
import math

from os.path import abspath, join

from numba import cuda 
from timeit import default_timer as timer  


WINDOW_SIZE = (24, 24)

@cuda.jit
def CalculateFeatures(featuresList,im_var,window_area,left,right,classifierThreshold):
      feature_sum = 0
      for feature in featuresList:
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

      if(feature_sum / im_var < float(classifierThreshold)):
          return left
      else:
          return right

#@cuda.jit    
def DetectFace(stages_np,features_np,my_stages_threshold_np,WINDOW_SIZE,img_draw,Stage_classifier_Count,AllWindows_np,AllWindow_Squared_np,face_indexes):
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
                classifiers_result = 0
                for index , classifiers in enumerate(stage):

                    if(index > Stage_classifier_Count[c]-1):
                        break

                    featureList = features_np[int(classifiers[1])]
                    for feature in features_np[int(classifiers[1])]:
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
                        classifiers_result += classifiers[2]
                    else:
                        classifiers_result += classifiers[3]

                       # CalculateFeatures(featuresList,im_var,window_area,left,right,classifierThreshold)
                #current_stage_result = sum(classifiers_result)
                if(classifiers_result < my_stages_threshold_np[c]):
                    face_found = False
                    #if c < 3:
                    #    break
                    #print(f"Stage {c} failed")
                    #print(f"stage threshold : {c}")
                    #print(f"stage current value : {classifiers_result}")
                   # break
            if face_found:
                face_indexes[index] = -1
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


#stages contains vector of classifiers
stages, features,my_stages_threshold,stage_classifiers_count = parse_haar_cascade_xml2(xml_path)

my_stages_threshold_np = np.array(my_stages_threshold , dtype=np.float32)
print(my_stages_threshold_np)
stages_np = np.array(stages, dtype= np.float32)
features_np = np.array(features, dtype=np.int32)
stage_classifiers_count_np = np.array(stage_classifiers_count,dtype=np.int32)


y_max, x_max = img_gray.shape
device = cuda.get_current_device()
threadsperblock = device.WARP_SIZE
block_dim = (threadsperblock, threadsperblock)
blockperthread = int(np.ceil(float((y_max- int(WINDOW_SIZE[0]) - 1)*(x_max- int(WINDOW_SIZE[0]) - 1))/threadsperblock))
#blockperthread = 1024


print(threadsperblock)
print(blockperthread)

# print(f'x_max : {x_max}')
# print(f'y_max : {y_max}')
plt.imshow(img_draw, cmap="gray")
plt.show()



start = timer()

scale = 1

GPU_stages = cuda.to_device(stages_np)
GPU_features = cuda.to_device(features_np)
GPU_my_stages_threshold_np = cuda.to_device(my_stages_threshold_np)
GPU_stage_classifiers_count = cuda.to_device(stage_classifiers_count_np)
GPU_WINDOW_SIZE =  cuda.to_device(WINDOW_SIZE)

print("start")



AllWindows = []
AllWindows_Squard = []
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
        AllWindows.append(window)
        AllWindows_Squard.append(window_squared)



print("finish")

AllWindows_np = np.array(AllWindows, dtype = np.int32)
AllWindow_Squared_np = np.array(AllWindows_Squard, dtype = np.int32)
GPU_AllWindows_np = cuda.to_device(AllWindows_np)
GPU_AllWindow_Squard = cuda.to_device(AllWindow_Squared_np)

faceIndex = np.zeros(int(np.ceil(float(y_max*x_max))))

GPU_face_indexes = cuda.to_device(faceIndex)

#DetectFace[blockperthread, threadsperblock](GPU_stages,GPU_features,GPU_my_stages_threshold_np,GPU_WINDOW_SIZE,GPU_stage_classifiers_count,GPU_AllWindows_np,GPU_AllWindow_Squard,GPU_face_indexes)


DetectFace(stages_np,features_np,my_stages_threshold_np,WINDOW_SIZE,img_draw,stage_classifiers_count,AllWindows,AllWindows_Squard,faceIndex)

faceIndex = GPU_face_indexes.copy_to_host()

print(faceIndex)


plt.imshow(img_draw, cmap="gray")
plt.show()
