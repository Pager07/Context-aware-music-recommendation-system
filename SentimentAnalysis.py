#!/usr/bin/env python
# coding: utf-8

# In[107]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# In[3]:


analyser = SentimentIntensityAnalyzer()


# In[123]:


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score['compound']


# In[132]:


def initSentimentScore():
    '''
    This function should be called to initialize the sentiment score
    '''
    return sentiment_analyzer_scores("I am feeling neutral!")


# In[125]:


def getSentence(emotion):
    '''
    Helper function of storeSentimentScore()
    '''
    return f'I am feeling {emotion}!'


# In[126]:


sentimentScores = []
def storeSentimentScore(emotionList):
    '''
       This function is called in 1 events
       1. Every frame 
    '''
    sentences = [getSentence(emotion) for emotion in emotionList]
    sentimentScoreList = [sentiment_analyzer_scores(sentence) for sentence in sentences]
    sentimentScores.extend(sentimentScoreList)

def reset():
    '''
    This function is called in 2 events
    1. User clicks get recommendation, thus start new tracking
    2. User click change user, thus start new tracking
    '''
    global sentimentScores
    sentimentScores = []


# In[131]:


def getSentimentScore():
    '''
        This function is called in 2 events
        1. User clicks get recommendation, thus start new tracking
        2. User clicks change user, thus start new tracking
    '''
    if len(sentimentScores) != 0:
        mean = sum(sentimentScores) / len(sentimentScores)
    else:
        mean = 0.0000
    return round(mean,3)

