import facedetectionCython
from xmlparser import parse_haar_cascade_xml
from os.path import abspath, join
from timeit import default_timer as timer  
import skimage.io as io
import cv2
import numpy as np

curr_dir = abspath(r'.')
#curr_dir = abspath(r'../../../.')

image_path = join(curr_dir, r"./images/faces/Ali.jpg")
#image_path = join(curr_dir, r"./images/faces/man1.jpeg")
#image_path = join(curr_dir, r"./images/Neutral/image0000767.jpg")



img_gray = io.imread(image_path, as_gray=True)
img_gray = 255 * img_gray
img_gray = cv2.resize(img_gray,(640, 480))
img_draw = img_gray.astype(np.uint8)
img_gray = img_gray.astype(np.uint64)
xml_path = join(curr_dir, r"./high-level/haar-cascades/haarcascade_frontalface_default.xml")
stages, features = parse_haar_cascade_xml(xml_path)

start = timer()
faces = facedetectionCython.getFaces(img_gray, stages, features)
print(timer() - start)

print(faces)