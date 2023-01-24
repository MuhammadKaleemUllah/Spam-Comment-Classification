#!/usr/bin/env python
# coding: utf-8

# # Spam Comments Detection using Python

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


# Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/images

# In[2]:


data = pd.read_csv("Youtube01-Psy.csv")
print(data.sample(5))


# In[3]:


data = data[["CONTENT", "CLASS"]]
print(data.sample(5))


# In[4]:


data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1:"Spam Comment"})
print(data.sample(5))


# # Bernoulli Naive Bayes
# 
# is one of the variants of the Naive Bayes algorithm in machine learning. It is very useful to be used when the dataset is in a binary distribution where the output label is present or absent. The main advantage of this algorithm is that it only accepts features in the form of binary values such as:
# 
# 1. True or False
# 2. Spam or Ham
# 3. Yes or No
# 4. 0 or 1

# In[7]:


x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[8]:


sample = "Check this out: https://thecleverprogrammer.com/" 
data = cv.transform([sample]).toarray()
print(model.predict(data))


# In[10]:


sample = "Lack of information!"
data = cv.transform([sample]).toarray()
print(model.predict(data))


# In[ ]:




