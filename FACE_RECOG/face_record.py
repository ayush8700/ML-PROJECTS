import cv2
import numpy as np

import os


cap=cv2.VideoCapture(0)

detector = cv2.CascadeClassifier()
name=input('enter your name: ')
frames=[]
outputs=[]

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

    if key == ord('c'):
    #    cv2.imwrite(name+'.jpg',frame)

        frames.append(gray.flatten())
        outputs.append([names])


X=np.array(frames)
Y= np.array(outputs)

data=np.hstack([Y,X])

f_names='face_data.npy'

if os.path.exists(f_names):
    old=np.load(f_names)
    data=np.vstack([old,data])

np.save(f_names,data)

cap.release()
cv2.destroyAllWindows()
