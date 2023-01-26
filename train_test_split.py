#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df = pd.read_csv("D:\Tahir\IBM all Lab\Machine learning\carprice.csv")
df


# In[3]:


df.columns


# In[4]:


df.drop(['Car Model','Unnamed: 4'],axis=1,inplace=True) 
df


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =10) # tesst_size = 0.2 means 20% of the data will be used for test

x_train


# In[7]:


from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(x_train,y_train)


# In[8]:


model.predict(x_test)     # the predicted value 


# In[9]:


y_test    # actual values 


# In[10]:


# both values are near to equal but not the same 


# In[11]:


model.score(x_test,y_test)   # accuracy 

