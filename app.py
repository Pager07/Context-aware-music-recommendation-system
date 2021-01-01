#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
from imutils.video import VideoStream
import time 
import imutils
import threading
import numpy as np
#Flask libraries

from flask import Flask,render_template,url_for,request, jsonify
from flask import Response


# In[2]:


#import the emotion detection file 

import FaceDetection

#import recommendation file
import Recommendation 

#import sentiment analysis file
import SentimentAnalysis


# #Reccomendation Model Factorization Machine 

# In[3]:


#trackObjs = Recommendation.getUserReccomendation('1' , 0.2)
#initialize current user to random user
currentUser = '29235188'
currentSentimentScore = SentimentAnalysis.initSentimentScore()


# In[4]:


def getUserRecommendation():
    global currentUser
    trackObjs = Recommendation.getUserReccomendation(currentUser , currentSentimentScore)
    return trackObjs


# In[5]:


#f = trackObjs[0]['features']


# In[6]:


#f


# In[7]:


#trackObjs


# #Flask Routing

# In[8]:


outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)


# In[9]:


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


# In[10]:


@app.route("/video_feed")
def video_feed():
    return Response(generate() ,
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# In[11]:


@app.route("/getRecommendation" , methods=['POST'])
def getRecommendation():
    '''
    This function updates the currentSentimentScore to  estimated sentimentScore.
    Restart tracking sentimentScore
    
    
    '''
    global currentSentimentScore
    
    #Getting the current score and tracks for the score 
    currentSentimentScore = SentimentAnalysis.getSentimentScore()
    trackObjs = getUserRecommendation()
    
    #Reseting the sentimentScores
    SentimentAnalysis.reset()
    currentSentimentScore = SentimentAnalysis.initSentimentScore()
    
    htmlSnippet = render_template('recommendation.html', trackObjs=trackObjs)
    data = jsonify({'datax':htmlSnippet})
    return data


# In[12]:


@app.route("/changeUser" , methods=['POST'])
def changeUser():
    global currentUser
    currentUser = Recommendation.getRandomUserID()
    #rest the sentiment score
    SentimentAnalysis.reset()
    currentSentimentScore = SentimentAnalysis.initSentimentScore()
    
    trackObjs = getUserRecommendation()
    htmlSnippet = render_template('recommendation.html', trackObjs=trackObjs)
    
    data = jsonify({'datax':htmlSnippet , 'currentUser':str(currentUser)})
    return data 


# #OpenCv

# In[13]:


def generate():
    global outputFrame, lock
    
    #loop over frames from the ouput stream
    while True:
        #wait until lock is acquired
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg" , outputFrame)
            
            #ensure the frame was successfully enooded
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


# In[14]:


def emotionDetection():
    global vs , outputFrame , lock
    while True:
        frame = vs.read()
        
        #frame , objectSet = ObjectDetection.objectDetection(frame)
        frame, emotionList = FaceDetection.getFaceROI(frame)
        SentimentAnalysis.storeSentimentScore(emotionList)
        
        frame = imutils.resize(frame, width=700)
        
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
        
    
    


# In[ ]:


# check to see if this is the main thread of execution
if __name__ == '__main__':
    t = threading.Thread(target=emotionDetection)
    t.daemon = True
    t.start()
    app.run(threaded = True, use_reloader=False)
vs.stop()

    


# In[ ]:




