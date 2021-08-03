#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump
from scipy import sparse


# In[2]:


X_train = sparse.load_npz('/Users/tingchen/Desktop/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_train.npz')


# In[3]:


num = 5

model = LatentDirichletAllocation(n_components = num, verbose=1)
model.fit(X_train)


# In[4]:


dump(model, f"/Users/tingchen/Desktop/Yelp_Cultural_Analysis/lda_runs/{num}-topics-lda.joblib")

