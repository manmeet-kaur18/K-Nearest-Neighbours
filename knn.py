#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[49]:


dfx=pd.read_csv('Diabetes_XTrain.csv')
dfy=pd.read_csv('Diabetes_YTrain.csv')
dfx2=pd.read_csv('Diabetes_Xtest.csv')


# In[41]:


plt.style.use('seaborn')
dfx


# In[42]:


X=dfx.values
print(X.shape)


# In[52]:


X_test=dfx2.values
print(X_test)
print(X_test.shape)


# In[44]:


print(Y.shape)


# In[45]:


Y=Y.reshape((-1,))


# In[46]:


print(Y.shape)


# In[47]:


print(X.shape)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show


# In[164]:


def distance(x1,x2):
    return np.sqrt((sum(x1-x2)**2))

def KNN(X,Y,queryPoint,k=3):
    vals=[]
    m=X.shape[0]
    for i in range(m):
        d=distance(queryPoint,X[i])
        vals.append((d,Y[i]))
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    
#     print(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
#     print(new_vals)
    max_freq_index= new_vals[1].argmax()
    pred=new_vals[0][max_freq_index]
    return pred


# In[187]:



y_predicted=[]
n=X_test.shape[0]
for i in range(n):
    pred=KNN(X,Y,X_test[i])
    y_predicted.append(pred)
#     appendFile.write(pred)
    print(pred)
# pred=KNN(X,Y,X_test[191])
# print(pred)
# import csv
# with open('diabetestestdata.csv','a',newline='') as file:
# #     for num in y_predicted:
#     write=csv.writer(file,delimiter='\n')
#     write.writecols([y_predicted])


# In[172]:


Y_predicted=np.asarray(Y_predicted)


# In[173]:


Y_predicted.shape


# In[174]:


n=Y.shape[0]
print(Y.shape[0])
# for i in range 


# In[175]:


# count=0
# for i in range(n):
#     if(Y[i]==Y_predicted[i]):
#         count+=1
# print(count)


# In[150]:


# accuracy=count/576


# In[176]:


# accuracy


# In[ ]:




