import cv2
import numpy as np


cap=cv2.VideoCapture(0)

detector = cv2.CascadeClassifier()

while True:

    ret,frame=cap.read()

    if ret :
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut=frame[y:y+h , x:x+w]
            fix=cv2.resize(cut,(100,100))
            gray = cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)

        cv2.imshow("MyScreen",frame)
        cv2.imshow("MyFace",gray)


    key=cv2.waitKey(1)

    if key == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()
