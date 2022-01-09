# import the opencv library
import cv2
from FaceDetectionTestWebCam import getFaces
from xmlparser import parse_haar_cascade_xml
from os.path import abspath, join
import facedetectionCython


curr_dir = abspath(r'.')
xml_path = join(curr_dir, r"./haarcascade_frontalface_default.xml")
stages, features = parse_haar_cascade_xml(xml_path)
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)
    #gray = cv2.equalizeHist(gray)
    
    #faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=2, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = facedetectionCython.getFaces(gray,stages,features)
    #print(faces)
    for (x,y,scale) in faces:
        # print(x,y,w,h)
        frame = cv2.rectangle(frame,(x,y),(x + 24*scale,y + 24*scale),(255,0,0),2)
        #roi_gray = gray[y:y*scale, x:x*scale]
        #roi_color = frame[y:y*scale, x:x*scale]
        #eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        #for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    
    # Display the resulting framee
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()