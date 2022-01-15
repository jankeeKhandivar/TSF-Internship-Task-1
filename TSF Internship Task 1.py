#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation : Data Science and Business Analytics Internship
# 
# 
# ## Task 1: Prediction using Supervised ML
# ### A Linear Regression task to predict the percentage of a student based on the number of study hours per day
# 
# 
# ## Author: Jankee Khandivar
# ### Batch: December - 2021

# #### Dataset URL: http://bit.ly/w-data

# ### Import necessary libraries in your script

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ### Reading data from remote link

# In[3]:


taskData = pd.read_csv("http://bit.ly//w-data")
taskData.head(10)


# ### Plotting the distribution of scores

# In[4]:


taskData.plot(x='Hours', y='Scores', style='o')
plt.title("Hours vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage obtained")
plt.show()


# #### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score. 

# ### Preparing the data

# In[5]:


a = taskData.iloc[:, :-1].values
b = taskData.iloc[:, 1].values
a_tr, a_te, b_tr, b_te = train_test_split(a,b)


# ### Training the algorithm

# In[6]:


reg = LinearRegression()
reg.fit(a_tr,b_tr)
print("Training complete")


# ### Plotting the regression line and test data

# In[7]:


line = reg.coef_*a + reg.intercept_

plt.scatter(a,b)
plt.plot(a,line)
plt.show()


# ### Accuracy of the Algorithm

# In[10]:


ac = reg.score(a_tr,b_tr)
print("Accuracy of linear regression is = {}".format(ac))


# ### Making Predictions

# In[16]:


print(a_te) # Testing data - In Hours
b_pred = reg.predict(a_te) # Predicting the scores


# In[17]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': b_te, 'Predicted': b_pred})  
df 


# ### Making predictions for the score when student studies for 9.25 hrs/day

# In[11]:


hr = [[9.25]]
pred = reg.predict(hr)
print(hr[0][0],"hours of study can lead to",pred[0],"marks")


# ### Evaluating the model

# In[18]:


print('Mean Absolute Error:', metrics.mean_absolute_error(b_te, b_pred)) 

