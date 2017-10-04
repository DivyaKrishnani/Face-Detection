import cv2
import sys

CascadePath ='opencv/data/haarcascades/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(CascadePath)

video_capture = cv2.VideoCapture(0)  #Sets video source as default webcam

while True:
    ret, frame = video_capture.read() #captures frame by frame

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
     )

#To draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Video detecting faces" ,frame)  #Displays resulting frame 
    if cv2.waitKey(1) & 0xFF == ord('q'):   #To exit q is to be pressed
        break

video_capture.release()
cv2.destroyAllWindows()
