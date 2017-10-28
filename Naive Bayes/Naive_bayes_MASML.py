
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy as sp
df1=pd.read_csv("C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\t20.csv")#rem to use double backslash
#booleans[]
dfx=df1[df1.Innings1Team==df1.Winner]
dfy= df1[df1.Innings1Team!=df1.Winner]
dfx['Winner']=0
dfy['Winner']=1


df1=pd.concat([dfx,dfy],ignore_index=True)
df1.replace(['Abu Dhabi','Adelaide','Bangalore','Birmingham','Cape Town'],[0,1,2,3,4], inplace=True)
original=df1
print(df1)
#df_replaced= analytics_df.replace(['Male','Female'],[0,1], inplace=True)


# In[2]:

del df1['Innings1Team'] #as we no longer need the name of the team
print (df1)


# In[3]:

import random
def split_dataset(df1,sp_ratio):
    orig= df1
    train_len=(int)(len(df1)*sp_ratio)
    train_set=[]
    while(len(train_set)<=train_len):
        index = random.randrange(len(df1))
        rowhere=(list)(df1[index:index+1].values.flatten())
        train_set.append(rowhere)
        df1.drop(df1.index[index],inplace=True)
    return [train_set,df1,orig]


# In[4]:

#now test is in the form of a dataframe. To get it to work in our functions, we convert it to a list
i=0;tester=[]
while(i<len(df1)):
    rowhere=(list)(df1[i:i+1].values.flatten())
    tester.append(rowhere)
    i=i+1


# In[5]:

def separateClass(dataset):
    separate={}
    for i in range(len(dataset)):
        element=dataset[i]
        if(element[-1] not in separate):#checking if last element is in separate
            separate[element[-1]]=[]
        separate[element[-1]].append(element)
    return separate


# In[6]:

def means(numList):
    return np.sum(numList)/float(len(numList))

def stdDev(numList):
    avg=means(numList)
    var=np.sum(np.power((x-avg),2)for x in numList)/float(len(numList))
    return np.sqrt(var)


# In[7]:

def summarize(dataset):#calculating mean and std dev for each attribute
    summ=[(means(x),stdDev(x))for x in zip(*dataset)]
    del summ[-1]
    return summ


# In[8]:

#now that we have created a method to summarize data, we must be able to summarize each attribute for each class. 
def summarizebyclass(dataset):
    sep= separateClass(dataset)
    summa={} #to store all values
    for classValue, instances in sep.items():
        summa[classValue]=summarize(instances)
    return summa


# In[9]:

#for calculating probability
def Probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# In[10]:

#multiplying all the class probabiities
def ClassProb(summaries, data):
    resultofprobs = {}
    for classValue, classSummary in summaries.items():
        resultofprobs[classValue] = 1
        for i in range(len(classSummary)):
            mean, stdev = classSummary[i]
            x = data[i]
            resultofprobs[classValue] *= Probability(x, mean, stdev)
    return resultofprobs


# In[11]:

#Implementation of the predict() function
def predict(summaries, data):
    probabilities = ClassProb(summaries, data)
    Label, ResultProb = None, -1
    for classValue, probability in probabilities.items():
        if Label is None or probability > ResultProb:  #to find the output with maximum probability
            ResultProb = probability
            Label = classValue
    return Label


# In[12]:

#now to get the list of predictions from the testSet
def getPredictions(dataset, test):
    predictions = []
    print("Len is")
    print(len(test))
    for i in range(len(test)):
        result = predict(dataset, test[i])
        predictions.append(result)
    return predictions


# In[13]:

def getAccuracy(testSet, predictions):
        correct = 0 #the number of correct answers predicted
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]: #if the result field of the testset matches the prediction made
                correct += 1
        return (correct/float(len(testSet))) * 100.0 #reporting it as a percentage


# In[14]:

train,test,original= split_dataset(original,0.8)
#summarizer= summarizebyclass(train)
print(train)



# In[15]:

summaries = summarizebyclass(train)


# In[16]:

predicter= getPredictions(summaries,tester)
print(predicter)


# In[17]:

accuracy = getAccuracy(tester, predicter)
print(accuracy)


# In[18]:

x_train=[]
y_train=[]
for i in range(len(train)):
    x_train.append([train[i][0],train[i][1]])
    y_train.append(train[i][-1])
x_test=[]
y_test=[]
for i in range(len(tester)):
    x_test.append([tester[i][0],tester[i][1]])
    y_test.append(tester[i][-1])


# In[20]:

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test)*100)

