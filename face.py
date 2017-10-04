import numpy as np
import cv2
import sys

CascadePath ='opencv/data/haarcascades/haarcascade_frontalface_default.xml'

ImagePath = sys.argv[1]  #To get image in which faces need to be identified


FaceCascade = cv2.CascadeClassifier(CascadePath)   #Creating Haar cascade

picture = cv2.imread(ImagePath)    #Reading Image
gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)   #Converting to gray scale


#To detect faces in the given picture
faces = FaceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)


#To draw a rectangle around the face detected
for (x, y, w, h) in faces:
    cv2.rectangle(picture, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Displays the picture with detected faces
cv2.imshow("Detected faces" ,picture)
cv2.waitKey(0)
