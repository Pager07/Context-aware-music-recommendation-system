#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import imutils
import time
import cv2
from pathlib import Path
from  fastai.vision import  *


# In[2]:


args = {"model" : './FaceDetectionModel/res10_300x300_ssd_iter_140000.caffemodel',
        "prototxt": './FaceDetectionModel/deploy.prototxt.txt',
        "confidence":0.5,
        "image":'./FaceDetectionModel/iron_chic.jpg',
         "emotion": ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']}


# #Emotion Recognition model

# In[3]:


learn = load_learner('./FaceDetectionModel')


# In[4]:


# test_img = open_image(args['image'])
# x = learn.predict(test_img)
# x[1].item()


# In[5]:


def getStringEmotion(x):
    '''
    Helper function for getRoiEmotion()
    Returns the emtion string that maps form pred tensor to args
    '''
    index = x[1].item()
    return args['emotion'][index]


# In[6]:


def getRoiEmotion(roi):
    '''
    Helper function for getFaceROI()
    Returns the emotion of string type in the ROI
    '''
    roi = cv2.cvtColor(roi , cv2.COLOR_BGR2RGB)
    cv2.resize(roi,(255,255))
    roi = Image(pil2tensor(roi, dtype=np.float32).div_(255))
    x = learn.predict(roi)
    roiEmotion = getStringEmotion(x)
    return roiEmotion


# #Face Localization model

# In[7]:


net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# In[8]:


image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))


# In[9]:


# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()


# In[10]:


def getRoi(frame, left,top,right,bottom):
    '''
    Helper function for getFaceROI
    Returns ROI(region of interest) 
    '''
    roi = frame[top:bottom , left:right,:]
    return roi


# In[11]:


def draw(image,startX,startY,endX,endY,confidence,roiEmotion):
    '''
    Helper function for faceDetection()
    Returns the image with bbox and probability  drawn
    '''
    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    text = text + '|' + roiEmotion
    labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(image, (startX, startY), (endX, endY),
                 (0, 0, 255), 1)
    
    startY = max(startY, labelSize[1])
    cv2.rectangle(image, (startX, startY - round(1.5*labelSize[1])),
        (startX + round(1.5*labelSize[0]), startY + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, text, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)


# In[12]:


def getFaceROI(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
     # loop over the detections
    imageCopy = image[:]
    emotionList = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #Emotion Start
            roi = getRoi(image , startX ,startY , endX , endY)
            if roi.shape[0]  > 0 and  roi.shape[1]  > 0:
                roiEmotion = getRoiEmotion(roi)
                emotionList.append(roiEmotion)
                #Emotion End 
                draw(imageCopy,startX,startY,endX,endY,confidence,roiEmotion)
    return imageCopy,emotionList
    


# In[ ]:




