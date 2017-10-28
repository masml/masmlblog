
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\housing_price.csv")
sf=df.values.tolist()
df.drop(df.columns[0],axis=1,inplace=True)
x=[]
y=[]
temp=[]
for i in range(len(sf)):
    for j in range(1,len(sf[i])-1):
        temp.append(sf[i][j])
    x.append(temp)
    temp=[]
    y.append(sf[i][-1])
#print(x)
#print(y)
#df


# In[2]:

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=1)


# In[3]:

from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,Y_train)
print(clf.predict(X_test))
print(clf.score(X_test,Y_test)*100)


# In[4]:

def find(data):#to find min and max of the given attributes
    minmax=[]
    for i in range(len(data[0])):
        r=[row[i] for row in data]
        min_val=min(r)
        max_val=max(r)
        minmax.append([min_val,max_val])
    return minmax


# In[5]:

import random
def split_dataset(data,sp_ratio):
    orig= data
    train_len=(int)(len(data)*sp_ratio)
    train_set=[]
    while(len(train_set)<=train_len):
        index = random.randrange(len(data))
        rowhere=(list)(data[index:index+1].values.flatten())
        train_set.append(rowhere)
        data.drop(data.index[index],inplace=True)
    test_set=data.values.tolist()
    return [train_set,test_set]


# In[6]:

def normalizer(data,minmax): #for data scaling
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j]=(data[i][j]-minmax[j][0])/(minmax[j][1]-minmax[j][0])
    return data


# In[7]:

def predict(row,coeff):
    pred=coeff[0]
    for i in range(len(row)-1):
        pred+=coeff[i+1]*row[i]
    return pred


# In[8]:

def coefficient_LR(train, learning_rate, n_epoch):
    coeff= [0.0 for i in range(len(train[0]))] # to make a list for the number of coefficient and initialise them to zero.
    for epoch in range(n_epoch):
        for row in train:
            pred=predict(row,coeff)
            error=pred-row[-1]
            coeff=coeff-learning_rate*error
            for i in range(len(row)-1):
                coeff[i+1]-=learning_rate*error*row[i]
    return coeff


# In[9]:

def linear_regression(train,test,learning_rate,n_epochs):
    predictions=[]
    coeff=coefficient_LR(train,learning_rate,n_epochs)
    for row in test:
        pred=predict(row,coeff)
        predictions.append(pred)
    return predictions


# In[10]:

train,test=split_dataset(df,0.8)
minmax_train=find(train)
minmax_test=find(test)
ntrain=normalizer(train,minmax_train)
ntest=normalizer(test,minmax_test)


# In[11]:

def accuracy(test,predictions):
    accuracy=0.0
    i=0
    for row in test:
        a=np.abs(test[i][-1]-predictions[i])
        i+=1
        accuracy+=a
    accuracy=(accuracy/len(predictions))*100
    return 100-accuracy


# In[12]:

learning_rate=0.01
n_epochs=500
predictions=linear_regression(ntrain,ntest,learning_rate,n_epochs)
npredictions=[]
for pred in predictions:
    pred=pred*(minmax_test[-1][1]-minmax_test[-1][0])+minmax_test[-1][0]
    npredictions.append(pred)
print(npredictions)
accuracy=accuracy(ntest,predictions)
print("\n")
print(accuracy)


# In[ ]:



