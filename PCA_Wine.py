#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
import seaborn as sns


# In[50]:


wine=pd.read_csv('C:/Users/prate/Downloads/Assignment/PCA/wine.csv')


# In[51]:


wine.head()


# In[52]:


wine.describe


# In[53]:


wine.isnull().sum()


# In[54]:


# Normalizing the numerical data 
wine_norm = scale(wine)


# In[55]:


wine_norm


# In[56]:


pca = PCA()
pca_values = pca.fit_transform(wine_norm)


# In[57]:


pca_values


# In[58]:


pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_norm)


# In[59]:


pca_values


# In[60]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# In[61]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[62]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")


# In[63]:


pca_values[:,0:1]


# In[64]:


# plot between PCA1 and PCA2 
x = pca_values[:,0:1]
y = pca_values[:,1:2]
#z = pca_values[:2:3]
plt.scatter(x,y)


# In[65]:


# Final Dataframe
final_df=pd.concat([wine['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df


# In[66]:


# Visualization of PCAs
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)


# In[67]:


#Checking with other Clustering Algorithms
# Hierarchical Clustering


# In[68]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[69]:


plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(final_df,'complete'))


# In[70]:


# Create Clusters
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters


# In[71]:


y=pd.DataFrame(hclusters.fit_predict(final_df),columns=['clustersid'])
y['clustersid'].value_counts()


# In[72]:


# Adding clusters to dataset
wine_c=wine.copy()
wine_c['clustersid']=hclusters.labels_
wine_c


# In[73]:


wine_c.tail()


# In[74]:


#K-Means Clustering


# In[75]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[76]:


model=KMeans(n_clusters=3) 
model.fit(final_df)

model.labels_ # getting the labels of clusters assigned to each row 


# In[77]:


# Assign clusters to the data set
wine_k=wine.copy()
wine_k['clusters3id']=model.labels_
wine_k


# In[78]:


wine_k['clusters3id'].value_counts()


# In[ ]:




