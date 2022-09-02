#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

gauss=pd.read_csv("gaussdata.csv")
gauss=gauss[['x','y']]
plt.scatter(gauss['x'], gauss['y'], marker='o')
plt.show()

heart=pd.read_csv("heart.csv",sep=' ')
plt.scatter(heart['x'], heart['y'], marker='o')
plt.show()


# In[2]:


from sklearn.cluster import DBSCAN
y1_db = DBSCAN(eps = 0.33, min_samples = 10).fit_predict(gauss)
plt.scatter(gauss['x'], gauss['y'], c=y1_db)
plt.show()

y2_db = DBSCAN(eps = 2, min_samples = 10).fit_predict(heart)
plt.scatter(heart['x'], heart['y'], c=y2_db)
plt.show()


# In[3]:


from sklearn.cluster import KMeans
y1_km = KMeans(n_clusters=3).fit_predict(gauss)
plt.scatter(gauss['x'], gauss['y'], c=y1_km)
plt.show()

y2_km=KMeans(n_clusters=2).fit_predict(heart)
plt.scatter(heart['x'], heart['y'], c=y2_km)
plt.show()


# In[4]:


#heart['z']=np.append(np.zeros(1000),np.ones(2000))
#gauss_x=gauss[['x','y']]
#gauss_y=gauss['z']


# In[5]:


from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm.fit(gauss)
y1_gmm = gmm.predict(gauss)
plt.scatter(gauss['x'], gauss['y'], c=y1_gmm)
plt.show()

gmm2=GaussianMixture(n_components=2)
gmm2.fit(heart)
y2_gmm=gmm2.predict(heart)
plt.scatter(heart['x'], heart['y'], c=y2_gmm)
plt.show()


# In[12]:


from pyclust import KMedoids
y1_kmedoids=KMedoids(n_clusters=3).fit_predict(np.array(gauss[['x','y']]))
plt.scatter(gauss['x'], gauss['y'], c=y1_kmedoids)
plt.show()

y2_kmedoids=KMedoids(n_clusters=2).fit_predict(np.array(heart))
plt.scatter(heart['x'], heart['y'], c=y2_kmedoids)
plt.show()


# In[13]:


from pyclust import BisectKMeans
y1_bkm=BisectKMeans(n_clusters=3).fit_predict(np.array(gauss[['x','y']]))
plt.scatter(gauss['x'], gauss['y'], c=y1_bkm)
plt.show()

y2_bkm=BisectKMeans(n_clusters=2).fit_predict(np.array(heart))
plt.scatter(heart['x'], heart['y'], c=y2_bkm)
plt.show()


# In[50]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import math
import random
import time
 
global MAX # 用于初始化隶属度矩阵U
MAX = 10000.0
 
global Epsilon  # 结束条件
Epsilon = 0.0000001
 
def print_matrix(list):
	""" 
	以可重复的方式打印矩阵
	"""
	for i in range(0, len(list)):
		print (list[i])
 
def initialize_U(data, cluster_number):
	"""
	这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
	"""
	global MAX
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U
 
def distance(point, center):

	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return math.sqrt(dummy)
 
def end_conditon(U, U_old):
    """
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
    global Epsilon
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):
		    if abs(U[i][j] - U_old[i][j]) > Epsilon :
			    return False
    return True
 
def normalise_U(U):
	"""
	在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
	"""
	for i in range(0, len(U)):
		maximum = max(U[i])
		for j in range(0, len(U[0])):
			if U[i][j] != maximum:
				U[i][j] = 0
			else:
				U[i][j] = 1
	return U
 
 
def fuzzy(data, cluster_number, m):
	# 初始化隶属度矩阵U
	U = initialize_U(data, cluster_number)
	# print_matrix(U)
	# 循环更新U
	while (True):
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
		# 计算聚类中心
		C = []
		for j in range(0, cluster_number):
			current_cluster_center = []
			for i in range(0, len(data[0])):
				dummy_sum_num = 0.0
				dummy_sum_dum = 0.0
				for k in range(0, len(data)):
    				# 分子
					dummy_sum_num += (U[k][j] ** m) * data[k][i]
					# 分母
					dummy_sum_dum += (U[k][j] ** m)
				# 第i列的聚类中心
				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j簇的所有聚类中心
			C.append(current_cluster_center)
 
		# 创建一个距离向量, 用于计算U矩阵。
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)
 
		# 更新U
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0, cluster_number):
    				# 分母
					dummy += (distance_matrix[i][j ] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / dummy
 
		if end_conditon(U, U_old):
			#print ("已完成聚类")
			break
 
	U = normalise_U(U)
	return U 

def de_randomise_data(data,order):
    new_data=[[] for i in range(0,len(data))]
    for index in range(0,len(order)):
        new_data[order[index]]=data[index]
        print(data[index])
        return new_data
    
if __name__ == '__main__':
	data= np.array(gauss[['x','y']])
	# 调用模糊C均值函数
	res_U = fuzzy(data , 3 , 2)
	y1_fckm=np.zeros(1500)
	for i in range(0,1500):
		y1_fckm[i]=np.argmax(res_U[i])
	# 计算准确率
	plt.scatter(gauss['x'], gauss['y'], c=y1_fckm)
	plt.show()
    
	data2= np.array(heart)
	# 调用模糊C均值函数
	res2_U = fuzzy(data2 , 2 , 2)
	y2_fckm=np.zeros(3000)
	for i in range(0,3000):
		y2_fckm[i]=np.argmax(res2_U[i])
	# 计算准确率
	plt.scatter(heart['x'], heart['y'], c=y2_fckm)
	plt.show()


# In[ ]:





# In[ ]:




