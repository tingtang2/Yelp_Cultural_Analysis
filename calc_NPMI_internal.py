#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from scipy import sparse
import torch
from tqdm import tqdm


# ## Get data

# In[2]:


collection_word = np.load("./lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/nzwcr.npy")
topic_word = np.load("./lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/nzw.npy")


# In[5]:


c3rLDA_nzwc = np.load("/Users/tingchen/Desktop/Yelp_Cultural_Analysis/lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/nzwc_ORIGINAL.npy")


# In[135]:


reg_assign = np.load("./lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/reg_assign_mat.npy")
indicies_locs_eth = np.load("./lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/indicies_locs_eth.npy")


# In[22]:


X_val = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_val.npz')
X_test = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_test.npz')
cc_val= np.load("/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/y_val.npy")
cc_test= np.load("/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/y_test.npy")

total_c = np.concatenate((cc_val, cc_test))
total_X = sparse.vstack([X_val, X_test])

ix = total_X.getnnz(1)>0
total_X = total_X[ix]
total_c= total_c[ix]


# In[136]:


reg_assign.shape


# In[173]:


reg_assign[0]


# In[175]:


indicies_locs_eth.shape


# In[177]:


max_reg_ind = np.max(reg_assign, axis=1) # number of regions assigned to each cuisine


# ## Rank top 10 words

# ### c3rLDA

# In[127]:


# top10_background = np.argsort(topic_word, axis=1)[:, -20:]

# intersect = np.sum(np(X_train[:, top10_background[0][:2]], axis=1))

# individuals = np.sum(X_train[:, top10_background[0][:2]], axis=0)

# pmi = np.log(10) + np.log(intersect) - np.log(individuals[0, 0]) - np.log(individuals[0, 1])
# npmi = -1 * pmi / (np.log(intersect) - math.log(10))


# In[5]:


def calculate_npmi(topics):
    npmi_means = []
    for i, topic in enumerate(topics):
        npmi_vals = []
        words = topic
        print("\rTopic", i, end="")
        for word_i, word1 in enumerate(words):
            for word2 in words[word_i+1:]:
                col1 = np.array(total_X[:, word1].todense() > 0, dtype=int)
                col2 = np.array(total_X[:, word2].todense() > 0, dtype=int)
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = np.sum(col1 * col2)
                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (np.log10(total_X.shape[0]) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(total_X.shape[0]) - np.log10(c12))
                npmi_vals.append(npmi)
                #print(str(word1), str(word2))

        npmi_means.append(np.mean(npmi_vals))
        
    print(np.mean(npmi_means))
    return np.mean(npmi_means)

def calculate_npmi_culture(topics, cult):
    npmi_means = []
    for i, topic in enumerate(topics):
        npmi_vals = []
        words = topic
        print("\rTopic", i, end="")
        for word_i, word1 in enumerate(words):
            for word2 in words[word_i+1:]:
                cult_ind = np.where(total_c == cult)[0]
                cult_X = total_X[cult_ind]
                
                col1 = np.array(cult_X[:, word1].todense() > 0, dtype=int)
                col2 = np.array(cult_X[:, word2].todense() > 0, dtype=int)
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = np.sum(col1 * col2)
                print(c1, c2, c12, cult_X.shape[0])
                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (np.log10(cult_X.shape[0]) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(cult_X.shape[0]) - np.log10(c12))
                npmi_vals.append(npmi)
                print(npmi)

        npmi_means.append(np.mean(npmi_vals))
    print("\n")
    print(np.mean(npmi_means))
    return np.mean(npmi_means)


# In[166]:


top10_background = np.argsort(topic_word, axis=1)[:, -10:]


# In[128]:


calculate_npmi(top10_background) # 20


# In[167]:


calculate_npmi(top10_background) # 10


# In[153]:


c3rLDA_nzwc.shape[2]


# In[150]:


c3rlda_culture_npmi = []

for c in range(c3rLDA_nzwc.shape[2]):
    c3rlda_culture_npmi.append(calculate_npmi_culture(np.argsort(c3rLDA_nzwc[:, :, c], axis=1)[:, -10:], c))
    
np.mean(c3rlda_culture_npmi)


# In[161]:


c3rlda_npmi = []

for c in range(c3rLDA_nzwc.shape[2]):
    c3rlda_culture_npmi.append(calculate_npmi(np.argsort(c3rLDA_nzwc[:, :, c], axis=1)[:, -10:]))
    


# In[ ]:


np.mean(c3rlda_npmi)


# In[165]:


np.mean(c3rlda_culture_npmi[28:])


# # npmi region

# In[180]:


region_npmis = []

for c in range(collection_word.shape[2]):
    for r in range(max_reg_ind[c]):
        region_npmis.append(calculate_npmi(np.argsort(collection_word[:, :, c, r], axis=1)[:, -10:]))


# In[181]:


np.mean(region_npmis)


# ## LDA

# In[142]:


np.where(total_c == 3)[0]


# In[98]:


from joblib import dump, load
model = load('./lda/lda.joblib') 


# In[107]:


model.components_


# In[168]:


top_10_words= np.argsort(model.components_, axis=1)[:, -10:]


# In[132]:


calculate_npmi(top_10_words) # 20


# In[170]:


calculate_npmi(top_10_words) # 10


# ## ccLDA

# In[109]:


ccLDA_nzw = np.load("./lda/ccLDA_nzw.npy")


# In[171]:


top_cc = np.argsort(ccLDA_nzw, axis=1)[:, -10:]


# In[126]:


calculate_npmi(top_cc) # 5 .221 - LDA


# In[172]:


calculate_npmi(top_cc) # 10


# In[134]:


calculate_npmi(top_cc) # 20


# In[112]:


ccLDA_nzwc = np.load("./lda/ccLDA_nzwc.npy")


# In[179]:


culture_npmi = []

for c in range(ccLDA_nzwc.shape[2]):
    culture_npmi.append(calculate_npmi(np.argsort(collection_word[:, :, c, r], axis=1)[:, -10:]))
    
np.mean(culture_npmi)


# In[151]:


cclda_culture_npmi = []

for c in range(ccLDA_nzwc.shape[2]):
    cclda_culture_npmi.append(calculate_npmi_culture(np.argsort(ccLDA_nzwc[:, :, c], axis=1)[:, -10:], c))
    
np.mean(cclda_culture_npmi)


# # Scholar NPMI on YELP

# In[3]:


npz = np.load("/home/tc2vg/scholar/output/beta.npz")


# In[6]:


import json

vocab = []
with open('/scratch/tc2vg/scholar_data/train.vocab.json', 'r', encoding='utf-8') as input_file:
    for line in input_file:
        vocab= json.loads(line, encoding='utf-8')


# In[5]:


data = np.load("/scratch/tc2vg/scholar_data/train.npz")


# In[6]:


beta = npz['beta']


# In[7]:


top10_scholar = np.argsort(beta, axis=1)[:, -10:]


# In[27]:


calculate_npmi(top10_scholar)


