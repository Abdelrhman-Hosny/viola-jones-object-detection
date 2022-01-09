# import the opencv library
import cv2
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import facedetectionCython
from xmlparser import parse_haar_cascade_xml
from os.path import abspath, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5') 
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


    faces = facedetectionCython.getFaces(gray,stages,features)
    for (x,y,scale) in faces:
        # print(x,y,w,h)
        frame = cv2.rectangle(frame,(x,y),(x + 24*scale,y + 24*scale),(255,0,0),2)
        roi_gray=gray[y:y+ 24*scale,x:x+ 24*scale]
        roi_gray=cv2.resize(roi_gray,(48,48))

        #Processes the image and adjust it to pass it to the model
        image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
    
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255

        #Get the prediction of the model
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]

        
        #Write on the frame the emotion detected
        cv2.putText(frame,emotion_prediction,(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    


    
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