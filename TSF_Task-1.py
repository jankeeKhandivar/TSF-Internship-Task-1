#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation : Data Science and Business Analytics Internship 
# 
# ## Task 1: Prediction using Supervised ML
# ### A Linear Regression task to predict the percentage of a student based on the number of study hours per day.
#  
# ## Author: Jankee Khandivar
# ### Batch: March - 2022

# #### Dataset URL: http://bit.ly/w-data

# ### Step:1  Import necessary libraries in your script
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ### Step:2 Reading data from remote link
# 

# In[2]:


taskData = pd.read_csv("http://bit.ly//w-data")
taskData.head(10)


# ### Step:3 Plotting the distribution of scores

# In[3]:


taskData.plot(x='Hours', y='Scores', style='o')
plt.title("Hours vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage obtained")
plt.show()


# #### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score. 

# ### Step:4 Preparing the data

# In[5]:


a = taskData.iloc[:, :-1].values
b = taskData.iloc[:, 1].values
a_tr, a_te, b_tr, b_te = train_test_split(a,b)


# ### Step:5 Training the algorithm

# In[6]:


reg = LinearRegression()
reg.fit(a_tr,b_tr)
print("Training complete")


# ### Step:6 Plotting the regression line and test data
# 

# In[8]:


line = reg.coef_*a + reg.intercept_

plt.scatter(a,b)
plt.plot(a,line)
plt.show()


# ### Step:7 Accuracy of the Algorithm

# In[9]:


ac = reg.score(a_tr,b_tr)
print("Accuracy of linear regression is = {}".format(ac))


# ### Step:8 Making Predictions

# In[11]:


print(a_te) # Testing data - In Hours
b_pred = reg.predict(a_te) # Predicting the scores


# In[12]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': b_te, 'Predicted': b_pred})  
df 


# ### Step:9 Making predictions for the score when student studies for 9.25 hrs/day

# In[13]:


hr = [[9.25]]
pred = reg.predict(hr)
print(hr[0][0],"hours of study can lead to",pred[0],"marks")


# ### Step:10 Evaluating the model

# In[14]:


print('Mean Absolute Error:', metrics.mean_absolute_error(b_te, b_pred)) 

