# coding: utf-8

# In[11]:


import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('C:\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalcatface.xml')

Id = input('Enter your id:')
sampleNum = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite(
            "F:/Courses/Data Science/face-recognising/dataset/user." + Id + '.' + str(sampleNum) + ".jpg",
            gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()
