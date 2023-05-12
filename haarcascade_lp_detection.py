import cv2
import numpy as np



def lpdetect2(img):
    frameWidth=840
    frameHeight=580
    nPlateCascade=cv2.CascadeClassifier("haarcascade_licence_plate.xml")
    minArea=500
    color=(0,128,0)

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates=nPlateCascade.detectMultiScale(imgGray,1.2,8)
    eps = 10
    imgs = img.copy()
    for(x,y,w,h) in numberPlates:
        area=w*h
        if area>minArea:
            img = imgs.copy()
            cv2.rectangle(img,(x,y),(x+w+eps,y+h+eps),(0,128,0),2)

            imgReg = img[y:y+h+eps,x:x+w+eps] #region of number plate
            cv2.imwrite("cim.jpg",imgReg)
            

    return img


