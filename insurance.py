
# coding: utf-8

# # Linear Regression using Least Square Method

# In[28]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading data
insurance_data=pd.read_csv('C:/Users/Anilkumar/Desktop/insurance.csv')


# In[3]:


insurance_data.head(15)


# In[29]:


print(insurance_data.shape)
insurance_data.head()


# In[30]:


#collect x & y
x=insurance_data['age'].values
y=insurance_data['bmi'].values


# In[31]:


#mean x & y
mean_x = np.mean(x)
mean_y = np.mean(y)


# In[32]:


#total no of values
m=len(x)


# In[33]:


#using the formula to calculate b1,b0
numer = 0
denom = 0
for i in range(m):
    numer += (x[i] - mean_x) * (y[i]-mean_y)
    denom += (x[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)
    print(b1, b0)


# In[34]:


#print coefficient
b1=np.mean(b1)
b0=np.mean(b0)
print(b1,b0)


# In[35]:


#plotting regression line
max_x = np.max(x)+100
min_x = np.min(x)-100


# In[36]:


#calculating line values x & Y
x=np.linspace(min_x,max_x,1400)
y=b0+b1*x


# In[37]:


#plotting line
plt.plot(x,y,color='#58b970',label='Regression line')
#plotting scatter plot
plt.scatter(x,y,c='#ef5423',label='scatter plot')
plt.xlabel('age')
plt.ylabel('bmi')
plt.legend()
plt.show()


# In[16]:


ss_t=0
ss_r=0
for i in range(m):
    y_pred=b0+b1*x[i]
    ss_t += (y[i]-mean_y)**2
    ss_t += (y[i]-y_pred)**2
    r2 = 1-(ss_r/ss_t)
    R2=np.mean(r2)
    print(R2)


# # Linear Regression model using SCIKIT-learn(Machine Learning) library

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = x.reshape((-1,1))
reg=LinearRegression()
reg=reg.fit(x,y)
y_pred=reg.predict(x)
R2_score=reg.score(x,y)
print (R2_score)

