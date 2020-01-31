#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
from itertools import chain, combinations



get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('Cars93.csv')


# In[3]:


data[:2]


# In[4]:


print(data.keys())


# In[5]:


Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])


# In[6]:


scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[61]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.3)

print(np.shape(Y_train), np.shape(X_train))


# In[62]:


regresion = sklearn.linear_model.LinearRegression()
regresion.fit(X_train, Y_train)


# combinaciones_indices

# In[63]:


print(regresion.coef_, regresion.intercept_)
print(regresion.score(X_train,Y_train), regresion.score(X_test,Y_test))


# In[64]:


lasso = sklearn.linear_model.Lasso(alpha=0.5)
lasso.fit(X_scaled, Y)
plt.scatter(X_scaled[:,3], Y)
plt.scatter(X_scaled[:,3], lasso.predict(X_scaled), marker='^')
plt.xlabel(columns[3])
plt.ylabel('Price')
print(lasso.coef_)
print(lasso.score(X_scaled, Y))


# In[ ]:





# In[ ]:





# In[58]:





# In[ ]:





# In[ ]:




