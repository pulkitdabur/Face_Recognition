# In[14]:


# detection
import cv2
import numpy as np
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("F:/Courses/Data Science/face-recognising/recognizer/trainningData.yml")
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while (True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (id == 1):
            id = "Pulkit"
        elif (id == 3):
            id = "ravi"
        elif (id == 5):
            id = "bhuvi"
        elif (id == 69):
            id = "sallu"
        elif (id==21):
            id = "gaurav sir"
        elif id==78:
            id = "maneesh sir"
        cv2.putText(img, str(id), (x, y + h), font, 3, 255)
    cv2.imshow("Face", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()

