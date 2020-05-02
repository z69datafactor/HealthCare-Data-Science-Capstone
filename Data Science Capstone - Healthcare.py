#!/usr/bin/env python
# coding: utf-8

# Problem Statement
# 
# NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) research creates knowledge about and treatments for the most chronic, costly, and consequential diseases.
# 
# The dataset used in this project is originally from NIDDK. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# 
# Build a model to accurately predict whether the patients in the dataset have diabetes or not.
# 
# 
# Dataset Description
# 
# The datasets consists of several medical predictor variables and one target variable (Outcome). Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and more.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', '')


# # Project Task: Week 1
# Data Exploration:

# Project Task: Week 1
#     
# Data Exploration:
#     
# 1. Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
# 
#     • Glucose
# 
#     • BloodPressure
# 
#     • SkinThickness
# 
#     • Insulin
# 
#     • BMI
#     
# 2. Visually explore these variables using histograms. Treat the missing values accordingly.
# 
# 3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables. 

# In[2]:


healthcare_data = pd.read_csv ('C:/Users/acer/Desktop/Data Warehousing/Healthcare - Diabetes Capstone/health care diabetes.csv')


# In[3]:


healthcare_data.head


# In[4]:


healthcare_data.describe()


# In[5]:


print("Standard Deviation of each variables are ==> ")
healthcare_data.apply(np.std)


# In[6]:


healthcare_data.isnull().any()


# In[7]:


healthcare_data.info()


# In[8]:


print("Standard Deviation of each variables are ==> ")
healthcare_data.apply(np.std)


# In[9]:


Positive = healthcare_data[healthcare_data['Outcome']==1]
Positive.head(5)


# In[10]:


healthcare_data['Glucose'].value_counts().head(7)


# In[11]:


plt.figure(figsize=(6,4),dpi=100)
plt.xlabel('Glucose Class')
healthcare_data['Glucose'].plot.hist()
sns.set_style(style='darkgrid')
print("Mean of Glucose level is :-", healthcare_data['Glucose'].mean())
print("Datatype of Glucose Variable is:",healthcare_data['Glucose'].dtypes


# In[ ]:


plt.hist(healthcare_data['Glucose'])


# In[ ]:


healthcare_data['BloodPressure'].value_counts().head(7)


# In[ ]:


plt.hist(healthcare_data['BloodPressure'])


# In[ ]:


healthcare_data['SkinThickness'].value_counts().head(7)


# In[ ]:


plt.hist(healthcare_data['SkinThickness'])


# In[ ]:


healthcare_data['BMI'].value_counts().head(7)


# In[ ]:


plt.hist(healthcare_data['BMI'])


# In[ ]:


healthcare_data.describe().transpose()


# # Project Task: Week 2

# Data Exploration:
#     
# 1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# 2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# 3. Perform correlation analysis. Visually explore it using a heat map.

# In[ ]:


plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)


# In[ ]:


Positive['BMI'].value_counts().head(7)


# In[ ]:


plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)


# In[ ]:


Positive['Glucose'].value_counts().head(7)


# In[ ]:


plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)


# In[ ]:


Positive['BloodPressure'].value_counts().head(7)


# In[ ]:


plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)


# In[ ]:


Positive['SkinThickness'].value_counts().head(7)


# In[ ]:


plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)


# In[ ]:


Positive['Insulin'].value_counts().head(7)


# In[ ]:


#Scatter plot


# In[ ]:


BloodPressure = Positive['BloodPressure']
Glucose = Positive['Glucose']
SkinThickness = Positive['SkinThickness']
Insulin = Positive['Insulin']
BMI = Positive['BMI']


# In[ ]:


plt.scatter(BloodPressure, Glucose, color=['b'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('BloodPressure & Glucose')
plt.show()


# In[ ]:


import seaborn as sns  


# In[ ]:


g =sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
                   hue="Outcome",
                   healthcare_data=healthcare_data);


# In[ ]:




