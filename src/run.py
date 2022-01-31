from os.path import join
from facedetection.xmlparser import parse_haar_cascade_xml
from facedetection.cython_code.get_faces import get_faces
import cv2

haar_directory = './haar_cascades/'
xml_file = 'haarcascade_frontalface_default.xml'

stages, features = parse_haar_cascade_xml(join(haar_directory, xml_file))

image_directory = './images/faces'
image_file = 'man1.jpeg'

frame = cv2.imread(join(image_directory, image_file))

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = get_faces(gray, stages, features)

for (x, y, x_m, y_m) in faces:
    frame = cv2.rectangle(
        frame, (x, y), (x_m, y_m), (255, 0, 0), 2
    )


# Display the resulting framee
cv2.imshow("frame", frame)
cv2.waitKey(0)
