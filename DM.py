#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import random
from sklearn.datasets import fetch_20newsgroups


# In[3]:


dataset = fetch_20newsgroups(download_if_missing=True,remove = ('headers', 'footers', 'quotes'))


# In[7]:


dataset


# In[9]:


dataset['DESCR']


# In[11]:


dataset['target_names']


# In[14]:


len(dataset['target'])


# In[13]:


dataset['filenames']


# In[18]:


unique_element,counts = np.unique(dataset['target'],return_counts=True)
print(unique_element)
print(counts)


# In[24]:



for element,count in zip(np.array(dataset['target_names']),counts):
    print(element,count)


# In[62]:


data_df = pd.DataFrame({'data':dataset.data,'target':dataset.target})
data_df.head()


# # 数据预处理: 去除停用词、数字、符号等

# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


# In[64]:


import re
import string

alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
data_df['data'] = data_df.data.map(alphanumeric).map(punc_lower)
data_df.head()


# In[66]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[73]:


texts = dataset.data # Extract text
target = dataset.target # Extract target


# In[74]:


target


# In[78]:


vectorizer = TfidfVectorizer(stop_words='english',max_features=1000)
X = vectorizer.fit_transform(texts)
print(X)


# In[83]:


idf_values = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
print(vectorizer.vocabulary_)


# In[84]:


number_of_clusters = 20


# In[85]:


model = KMeans(n_clusters=number_of_clusters, 
               init='k-means++', 
               max_iter=100, 
               n_init=1)  


model.fit(X)


# In[86]:


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


# In[87]:


for i in range(number_of_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])


# In[88]:


y=model.predict(X)
y


# In[91]:


from sklearn import metrics
ch = metrics.calinski_harabasz_score(X.toarray() ,y)
ch


# In[92]:


sc=metrics.silhouette_score(X,y)
sc


# In[93]:


from sklearn.decomposition import PCA


# In[94]:


from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(shuffle=True, random_state=777, remove=('headers', 'footers', 'quotes'))


# In[95]:


from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(shuffle=True, random_state=777, remove=('headers', 'footers', 'quotes'))


# In[96]:



from sklearn.feature_extraction.text import TfidfVectorizer

data_samples = news.data[:1000]
data_target = news.target[:1000]
data_class = news.target_names
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100, stop_words='english')
TFIDF = tfidf_vectorizer.fit_transform(data_samples)


# In[97]:


print(TFIDF.toarray()[0:2])


# In[99]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(TFIDF.toarray())


# In[100]:


import matplotlib.cm as cm

PCA_TFIDF = pca.transform(TFIDF.toarray())
print(PCA_TFIDF.shape)

plt.figure(figsize=(10,10))
plt.scatter(PCA_TFIDF[:,0], PCA_TFIDF[:,1], c=data_target)
plt.show()


# In[ ]:






