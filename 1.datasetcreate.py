import cv2
import numpy as np
import time
import os

cascadepath = r"haarcascades\haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadepath)
cam = cv2.VideoCapture(0)

count = 0
id = input('Enter the id:')
while True:
    
    _,frame = cam.read()
    faces = faceDetect.detectMultiScale(frame,1.1,3)

    for x,y,w,h in faces:
        writepath = os.path.join('DATA','user')
        cv2.imwrite(f"{writepath}{id}.{str(count)}.jpg", frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        count += 1
        print(count)

    cv2.imshow("FACE",frame)
    if cv2.waitKey(1) & count==200:
        break
    time.sleep(0.2)

cam.release()
cv2.destroyAllWindows()