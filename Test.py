import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math



cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier=Classifier("Model/Keras_model.h5","Model/labels.txt")

offset=20
imgSize=300
counter=0

labels=["A","B","C"]

folder="Data/C"

while True:
    success,img=cap.read()
    imgOutPut=img.copy()
    hands, img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropshape=imgCrop.shape


        aspectratio=h/w
        if aspectratio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[0:imgResizeShape[0], wGap:wCal+wGap] = imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            print(prediction)
            print(index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction)
            print(index)

        cv2.putText(imgOutPut,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
        cv2.rectangle(imgOutPut,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,255,255),4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key=cv2.waitKey(1)

