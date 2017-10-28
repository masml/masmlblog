
# coding: utf-8

# In[1]:

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

#Read csv file
#data_frame = pd.read_csv('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.csv')
#Read excel file
data_frame_1 = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.xlsx', "nyc_1")
data_frame_2 = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.xlsx', "nyc_2")


# In[3]:

#Web Scraping
#Libraries- html5lib, lxml, BeautifulSoup4
import html5lib
from lxml import etree
url = 'https://en.wikipedia.org/wiki/States_and_union_territories_of_India'
df_web = pd.read_html(url, header = 0, flavor='html5lib')


# In[4]:

df_web[2].set_index('Vehicle code')


# In[5]:

df = data_frame_2.set_index('EST')
df


# In[6]:

#We need to remove all the missing data from our dataset. 
#So we have 3 options
#1. Remove all the unneccessary elements
#2. Replace the unavailable elements with some base value
#3. Improve upon the previous by interpolation

#Option 1
method_1 = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.xlsx', "nyc_2",
                         na_values={'Events':['n.a'], 'WindSpeedMPH':[-1]})
'''
There are 3 parameters: pd.read_excel('file_path','sheet_name','na_values')
na values are the missing values in the dataset. Your dataset may have different messy values like 'n.a', 'not available' 
(for string types) and negative values for numeric types (not necessarily like penalty score is a correct negative value)
We are specifying all the messy values using a dictionary in which the column name and list of messy values is specified
'''

#Removes all those na_values
method_1.dropna()


# In[7]:

#Option 2
method_2 = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.xlsx', "nyc_2",
                         na_values={'Events':['n.a'], 'WindSpeedMPH':[-1]})
'''
There are 3 parameters: pd.read_excel('file_path','sheet_name','na_values')
na values are the missing values in the dataset. Your dataset may have different messy values like 'n.a', 'not available' 
(for string types) and negative values for numeric types (not necessarily like penalty score is a correct negative value)
We are specifying all the messy values using a dictionary in which the column name and list of messy values is specified
'''

#If we do not want to remove the na values as in the previous step, we can replace the na values with some chosen base value 
method_2.fillna({
        'WindSpeedMPH':0,
        'Events':"Sunny",
        'CloudCover':3
    })


# In[8]:

'''
If you do not want to specify your own base values and want it to be based on certain criterion
like same as the previous or the next day with an available values, so we need to specify that as well
df.fillna(method="method_name")
You have a lot of options like ffill to carry forward the previous value, bfill (just the opposite) and so on
To view the complete documentation, you can refer to
" http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html"
'''
method_2.fillna(method="ffill", inplace=True)
'''
We specified an extra parameter inplace. What it does is that it performs all the changes on the original dataframe itself which
is by default false. 
method_2.fillna(method="ffill")
This would have made no changes in the original dataframe, i.e., method_2 itself.
'''
method_2


# In[9]:

#Option 3
method_3 = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\nyc_weather.xlsx', "nyc_2",
                         na_values={'Events':['n.a'], 'WindSpeedMPH':[-1]})
method_3.interpolate()
#By default, interpolate implements a linear approach to fill the data.(9th and 10th value changed)
#You can specify the method by introducing the method parameter which can be done as follows:
method_3.interpolate(method="quadratic")
#Note the change in the values due to linear and quadratic approach. You have a lot of different methods which
#can be referred in the provided link above


# In[10]:

df= method_2.groupby('Events')
df.get_group('Snow')


# In[11]:

df= method_2.groupby('Events')
for events, events_df in df:
    print(events)
    print(events_df)
get_ipython().magic('matplotlib inline')
df.plot()


# In[12]:

#Concating two parts of a dataframe
df1=data_frame_1
df2=data_frame_2
merge_df = pd.merge(df1,df2,on="EST")
merge_df
#It takes redundant columns into consideration too, axis=1 implies vertical concatentation


# In[13]:

concat_frame_1 = pd.read_csv('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\weather2.csv')
concat_frame_2 = pd.read_csv('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\weather3.csv')
#concated_df= pd.concat([concat_frame_1,concat_frame_2],ignore_index=True)
concated_df= pd.concat([concat_frame_1,concat_frame_2],keys=["Set_1","Set_2"])
#By default, concat retains the index of the original data frame, so you can either seperate the groups by providing a list of keys
#or to have a continuous indexing, just apply ignore_index=True
concated_df


# In[14]:

cross_tab_df = pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\survey.xls')
pd.crosstab([cross_tab_df.Nationality,cross_tab_df.Sex],cross_tab_df.Handedness,margins=False,normalize="index")
#check the index parameter


# In[15]:

df= pd.read_csv('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\weather3.csv')
print(df)
#Make sure
transform_df_1= df.pivot(index="date", columns="city")
print(transform_df_1)
transform_df_2= df.pivot_table(index="date", columns="city", margins=True, aggfunc='sum')
print(transform_df_2)
df.stack(level=0)


# In[16]:

stack_df= pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\stocks_3_levels.xlsx', header=[0,1,2])
stack_df
stack_df.stack()
#There is a level parameter which is by default set to the innermost level (in this case= 2)


# In[17]:

#Make sure if you are changing value, you may encounter some NaN values since the columns are divided 
#and choosing a higher level may not have all the values
stack_df.stack(level=1)


# In[21]:

#Let us look at the general information we can get about a given data set
analytics_df= pd.read_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\survey.xls')
#Shape shows the dimensions of the dataset
print(analytics_df.shape)
#Describe shows the overall statistics of the dataset like mean, standard deviation, etc
print(analytics_df.describe())
#Replace is used to replace the data with some other value of your choice
df_replaced= analytics_df.replace(['Male','Female'],[0,1], inplace=True)
df_replaced= analytics_df.replace(['Left','Right'],[0,1])
'''analytics_df.replace({
        'Sex':"[A-Za-z]"
    },"lol",regex=True)'''
df_replaced


# In[19]:

final= stack_df.stack()
#Now that we have a final dataset for use, let us write it for future use.
#The write is similar to the read and we can write in various formats like csv,xls,etc.
final.to_csv('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\processed_dataset.csv')
final.to_excel('C:\\Users\\hp\\Desktop\\MAS ML\\datasets\\processed_dataset.xlsx')

