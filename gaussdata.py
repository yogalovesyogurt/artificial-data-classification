#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
#matplotlib inline
X1, y1=datasets.make_circles(n_samples=1250, factor=.65,noise=.05)
X2, y2 = datasets.make_blobs(n_samples=250, n_features=2, centers=[[1.2,-1.2]], cluster_std=[[.1]],random_state=9)
X = np.concatenate((X1, X2))
convex_x=X[:,0]
convex_y=X[:,1]
column_x = pd.Series(convex_x, name='x')
column_y = pd.Series(convex_y, name='y')
convex=pd.DataFrame({'x':convex_x,'y':convex_y})
convex.to_csv("convex.csv", index=False, sep=' ')

plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()


# In[3]:


from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[4]:


y_pred = DBSCAN(eps = 0.12, min_samples = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[5]:


gauss=pd.read_csv("gaussdata.csv")
plt.scatter(gauss['x'], gauss['y'], c=gauss['z'])
plt.show()


# In[22]:


gauss_y = DBSCAN(eps = 0.33, min_samples = 10).fit_predict(gauss)
plt.scatter(gauss['x'], gauss['y'], c=gauss_y)
plt.show()


# In[ ]:




