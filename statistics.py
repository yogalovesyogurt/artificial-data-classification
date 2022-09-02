#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# In[71]:


x = np.vstack((X_reduced[:, 0],X_reduced[:, 1]))
x = x.transpose()
x


# In[86]:


y_pred = DBSCAN(eps = 1).fit_predict(x)
plt.scatter(X_reduced[:, 0],X_reduced[:, 1],c=y_pred)
plt.show()


# In[73]:


from sklearn import datasets
Iris_df = datasets.load_iris()
lris_df = datasets.load_iris() 
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=lris_df.target)
plt.show()


# In[74]:


from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

lris_df = datasets.load_iris() 
#挑选出前两个维度作为x轴和y轴，你也可以选择其他维度

X_reduced = PCA(n_components=2).fit_transform(iris.data)
#这里已经知道了分3类，其他分类这里的参数需要调试
model = KMeans(n_clusters=3) 
#训练模型
model.fit(lris_df.data) 
#选取行标为100的那条数据，进行预测
prddicted_label= model.predict([[6.3, 3.3, 6, 2.5]]) 
#预测全部150条数据
all_predictions = model.predict(lris_df.data) 
#打印出来对150条数据的聚类散点图
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=all_predictions)
plt.show()


# In[75]:


model2 = KMeans(n_clusters=3,init='kmedoids(PAM,Partitioning Around Medoids)') 
model.fit(lris_df.data) 
#选取行标为100的那条数据，进行预测
prddicted_label= model.predict([[6.3, 3.3, 6, 2.5]]) 
#预测全部150条数据
all_predictions = model.predict(lris_df.data) 
#打印出来对150条数据的聚类散点图
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=all_predictions)
plt.show()


# In[76]:


import numpy as np

def FCM(X, c_clusters=3, m=2, eps=10):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X), np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
        
        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x-c, 2)
        
        new_membership_mat = np.zeros((len(X), c_clusters))
        
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat =  new_membership_mat
    return np.argmax(new_membership_mat, axis=1)


# In[77]:


from sklearn import datasets

iris = datasets.load_iris()


# In[78]:


def evaluate(y, t):
    a, b, c, d = [0 for i in range(4)]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j] and t[i] == t[j]:
                a += 1
            elif y[i] == y[j] and t[i] != t[j]:
                b += 1
            elif y[i] != y[j] and t[i] == t[j]:
                c += 1
            elif y[i] != y[j] and t[i] != t[j]:
                d += 1
    return a, b, c, d

def external_index(a, b, c, d, m):
    JC = a / (a + b + c)
    FMI = np.sqrt(a**2 / ((a + b) * (a + c)))
    RI = 2 * ( a + d ) / ( m * (m + 1) )
    return JC, FMI, RI

def evaluate_it(y, t):
    a, b, c, d = evaluate(y, t)
    return external_index(a, b, c, d, len(y))


# In[79]:


test_y = FCM(iris.data)


# In[80]:


evaluate_it(iris.target, test_y)


# In[81]:



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_y, cmap=plt.cm.Set1)


# In[110]:


from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=3).fit(x)
labels = gmm.predict(x)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],c=labels, s=40, cmap='viridis');

probs = gmm.predict_proba(x)
print(probs[101:110].round(2))


# In[ ]:




