#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


plt.style.use('ggplot')
sns.color_palette("tab10")
sns.set(context='notebook', style='darkgrid',font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[20,10]
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[3]:


df=pd.read_csv('Desktop/news.csv')
print(df.shape)
df.head()


# In[4]:


labels=df.label
labels.head()


# In[5]:


target=df.label.value_counts()
target


# In[9]:


sns.countplot(df.label)
plt.title('the number of news fake/real');


# In[10]:


X_train,X_test,Y_train,Y_test=train_test_split(df['text'], labels, test_size=0.3, random_state=6)


# In[12]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.6)
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)


# In[14]:


pac=PassiveAggressiveClassifier(max_iter=60)
pac.fit(tfidf_train,Y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(Y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[16]:


confusion_matrix(Y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




