import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
servo = GPIO.PWM(23, 180)
servo.start(5)
buka = float(10) / 10.0 + 2.5
tutup = float(170) / 10.0 + 2.5

faceDetect=cv2.CascadeClassifier \
          ('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.createLBPHFaceRecognizer();
rec.load("trainningData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_ \
                     COMPLEX_SMALL,3,1,0,2)
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="kiki"
            servo.ChangeDutyCycle(buka)
        elif(id==2):
            id="uni"
            servo.ChangeDutyCycle(tutup)
        elif(id==3):
            id="aa"
            servo.ChangeDutyCycle(tutup)
        cv2.cv.PutText(cv2.cv.fromarray(img), \
                       str(id),(x,y+h), font,225);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows
