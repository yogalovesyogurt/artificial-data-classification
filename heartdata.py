#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import random

def random_list(start,stop,length):
    random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
    return random_list
n=500
x1=np.array(random_list(-6,6,n))
x2=np.array(random_list(-12,12,2*n))


# In[17]:


h1=np.array(random_list(0,2.5,n))
h2=np.array(random_list(0,2.5,2*n))
y1 = random.uniform(0.618*np.abs(x1) - 0.8* np.sqrt(36-x1**2),0.618*np.abs(x1) - 0.8* np.sqrt(36-x1**2)+h1)
y2 = random.uniform(0.618*np.abs(x1) + 0.8* np.sqrt(36-x1**2)-h1,0.618*np.abs(x1) + 0.8* np.sqrt(36-x1**2))
y3 = random.uniform(0.618*np.abs(x2) - 0.8* np.sqrt(144-x2**2)-h2,0.618*np.abs(x2) - 0.8* np.sqrt(144-x2**2)+h2)
y4 = random.uniform(0.618*np.abs(x2) + 0.8* np.sqrt(144-x2**2)-h2,0.618*np.abs(x2) + 0.8* np.sqrt(144-x2**2))
plt.scatter(x1, y1, color = 'r')
plt.scatter(x1, y2, color = 'r')
plt.scatter(x2, y3, color = 'b')
plt.scatter(x2, y4, color = 'b')
plt.show()


# In[18]:


x=np.hstack((x1,x1,x2,x2)).T
y=np.hstack((y1,y2,y3,y4)).T
X=np.vstack((x,y)).T
np.shape(X)

import pandas as pd
column_x = pd.Series(x, name='x')
column_y = pd.Series(y, name='y')
heart=pd.DataFrame({'x':x,'y':y})
heart.to_csv("heart.csv", index=False, sep=' ')


# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:




