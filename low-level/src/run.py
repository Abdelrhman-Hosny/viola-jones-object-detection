from facedetection.xmlparser import parse_haar_cascade_xml
from facedetection.cython_code.get_faces import get_faces
import cv2

xml_path = "/home/hos/Coding/image-project/fer/low-level/haar_cascades/haarcascade_frontalface_default.xml"
stages, features = parse_haar_cascade_xml(xml_path)

frame = cv2.imread("/home/hos/Coding/image-project/fer/images/faces/man1.jpeg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = get_faces(gray, stages, features)

for (x, y, x_m, y_m) in faces:
    frame = cv2.rectangle(
        frame, (x, y), (x_m, y_m), (255, 0, 0), 2
    )


# Display the resulting framee
cv2.imshow("frame", frame)
cv2.waitKey(0)
