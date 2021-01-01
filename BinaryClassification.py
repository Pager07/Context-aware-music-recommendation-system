#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyfm import pylibfm
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


# In[4]:


path = Path('/Users/sandeep/Desktop/Homework2019/Recommender/expressionCapture/nowplayingRS')


# In[5]:


df = pd.read_csv(path/'nowplayingFinal.csv')
df = df.drop(columns=['Unnamed: 0'])


# In[8]:


df.head()


# In[7]:


df['rating'] = np.where(df['rating']>= 0.5, 1, 0)


# In[9]:


Y_df = df['rating']


# In[10]:


X_df = df.drop(columns=['rating'])


# In[11]:


X_df['user_id'] = X_df['user_id'].astype('str')


# In[15]:


X_df = X_df.to_dict('records')


# In[16]:


X = X_df
Y = Y_df.values


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state=42)


# In[18]:


v = DictVectorizer()


# In[19]:


X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)


# In[22]:


fm = pylibfm.FM(num_factors=50, num_iter=70, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")


# In[23]:


fm.fit(X_train, y_train)


# In[24]:


pickle.dump(fm, open( "fittedmodel2.pickle", "wb" ), -1) 


# In[35]:


pred = fm.predict(X_test)


# In[ ]:


pred = np.where(pred>=0.5, 1 , 0)


# In[45]:


print ("Test MAE: %.4f" % mean_absolute_error(y_test,pred))


# In[46]:


print ("Test Precision: %.4f" % precision_score(y_test,pred))


# In[51]:


print ("Test Recall: %.4f" % recall_score(y_test,pred))


# In[53]:


tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()


# In[54]:


tp


# In[55]:


fn


# In[56]:


fp


# In[57]:


tn


# In[58]:


len(pred)


# In[59]:


confusion_matrix(y_test, pred)


# In[ ]:




