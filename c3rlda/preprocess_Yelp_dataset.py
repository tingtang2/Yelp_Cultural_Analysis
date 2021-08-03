#!/usr/bin/env python
# coding: utf-8

# # Analysing Culture in Yelp Reviews

# Ting Chen

# In[1]:


import json

import re
import stanza

import numpy as np
from matplotlib import pyplot as plt

import nltk
import tomotopy as tp

from gensim import corpora, matutils
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump, load

from scipy import sparse


# ## 1. Load in data

# In[2]:


# Ethnic cuisines we have chosen to observe

classifier_keys = ['Chinese', 'American (Traditional)', 'Asian Fusion',
        'Italian', 'Mexican', 'Japanese','Indian', 'French']


# In[17]:


# Load in reviews based on ethnicity from previous json dump

classifier_reviews = {}

with open('classifier_data.json', 'r') as file:
    classifier_reviews = json.load(file)


# In[2]:


all_eth_reviews = {}

with open('all_eths_reviews.json', 'r') as file:
    all_eth_reviews = json.load(file)


# In[4]:


all_eth_reviews['Chinese'][1]


# ## figure out date range

# In[5]:


dates = []
for key, value in all_eth_reviews.items():
    for val in value:
        dates.append(val['date'])


# In[7]:


sort = sorted(dates)


# In[8]:


sort[0]


# In[9]:


sort[-1]


# In[3]:


# down sample
too_small = []
for key, value in all_eth_reviews.items():
    if len(value) < 15000:
        too_small.append(key)


# In[4]:


for key in too_small:
    del all_eth_reviews[key]


# In[10]:


del all_eth_reviews['Canadian (New)']


# In[8]:


del all_eth_reviews['Ethnic Food']


# In[12]:


del all_eth_reviews['Asian Fusion']


# In[7]:


del all_eth_reviews['Halal']


# In[9]:


del all_eth_reviews['New Mexican Cuisine']


# In[6]:


del all_eth_reviews['American (New)']


# In[7]:


del all_eth_reviews['American (Traditional)']


# In[8]:


del all_eth_reviews['Irish']


# In[9]:


del all_eth_reviews['Cantonese']


# In[23]:


sum([len(x) for x in all_eth_reviews.values()])


# In[13]:


for key, value in all_eth_reviews.items():
    print(key + str(':'), len(value))


# In[14]:


len(all_eth_reviews.keys())


# Save keys

# In[25]:


with open('all_eths_minus_american_keys.json', 'w',encoding='utf-8') as outfile:
    json.dump(list(all_eth_reviews.keys()), outfile, ensure_ascii=False, indent=4)


# Make this a ternary classification for now to speed up training and processing

# In[18]:


ternary_keys = ['Chinese', 'Italian', 'Mexican']


# In[19]:


ternary_reviews = {}

for key in ternary_keys:
    ternary_reviews[key] = classifier_reviews[key]


# In[7]:


for key, value in ternary_reviews.items():
    print(key + str(':'), len(value))


# ## Find num of words and duplicates

# In[10]:


for key, value in all_eth_reviews.items():
    print(key, sum([len(review['text'].split()) for review in value]))


# In[11]:


reviews = []

for key, value in all_eth_reviews.items():
    for val in value:
        reviews.append(val['review_id'])


# In[13]:


len(set(reviews))


# Get locations for each review

# In[20]:


businesses = []
for line in open('yelp_dataset/yelp_academic_dataset_business.json', 'r'):
    businesses.append(json.loads(line))


# In[21]:


bus_table = {}

for bus in businesses:
    bus_table[bus['business_id']] = (bus['latitude'], bus['longitude'])


# In[22]:


locations = []
for key, value in all_eth_reviews.items():
    for review in value:
        bus_id = review['business_id']
        locations.append(bus_table[bus_id])


# In[23]:


len(locations) # All eths including american


# In[24]:


len(set(locations))


# In[30]:


len(locations)


# In[10]:


locations[1]


# In[31]:


locs = np.array(locations)


# In[22]:


with open('locs_unshuffles.json', 'w',encoding='utf-8') as outfile:
    json.dump(locations, outfile, ensure_ascii=False, indent=4)


# In[32]:


with open('locs_all_eths_minus_american.json', 'w',encoding='utf-8') as outfile:
    json.dump(locations, outfile, ensure_ascii=False, indent=4)


# In[3]:


locs = []

with open('locs_all_eths_minus_american.json', 'r') as infile:
    locs = json.load(infile)


# ## Exploring geographic distribution of cuisines

# In[14]:


all_eths = []

with open('all_eths.txt', 'r') as infile:
    all_eths = infile.read().split('\n')


# In[16]:


all_eths= all_eths[:-1]


# In[28]:


for eth in all_eths:
    locations = []

    for bus in businesses:
        if eth in str(bus['categories']):
            locations.append((bus['latitude'], bus['longitude']))
            
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    ax = world[world.continent == 'North America'].plot(
        color='white', edgecolor='black', figsize = (30, 30))


    df = pd.DataFrame(locations, columns= ['lat', 'long'])
    val = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.long, df.lat))

    val.plot(ax=ax, color='red', edgecolor='blue')

    plt.savefig('/Users/tingchen/Desktop/Yelp_Cultural_Analysis/lda/plots_eths/' + eth + '.png', bbox_inches='tight')
    plt.close()


# In[ ]:


gdf_regions = {}

for key in regions_locs.keys():
    df = pd.DataFrame(regions_locs[key], columns= ['lat', 'long'])
    gdf_regions[key] = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.long, df.lat))


# In[6]:


len(locations)


# In[15]:


import pandas as pd
import geopandas
import matplotlib.pyplot as plt


# In[13]:


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

ax = world[world.continent == 'North America'].plot(
    color='white', edgecolor='black', figsize = (30, 30))


colors= ['red', 'blue', 'green', 'purple', 'orange' ] #, 'yellow', 'pink', 'grey', 'brown', 'black']

df = pd.DataFrame(locations, columns= ['lat', 'long'])
val = geopandas.GeoDataFrame(
df, geometry=geopandas.points_from_xy(df.long, df.lat))

val.plot(ax=ax, color=colors[0])

plt.show()


# ## 2. Preprocess

# Example review text:

# In[8]:


ternary_reviews['Mexican'][233]


# In[ ]:





# For now best results are nltk tokenized and punctation removed

# In[20]:


stopwords = set(nltk.corpus.stopwords.words('english'))

classifier_text = {k:[] for k in ternary_keys}

for key, value in ternary_reviews.items():
    print(key)
    for i in range(int(len(value))):
        if i % 10000 == 0:
            print(i)
            
        doc = nltk.tokenize.word_tokenize(ternary_reviews[key][i]['text'])
        
        text = [token for token in doc if re.match("[a-zA-Z0-9']+", token)]
        
        classifier_text[key].append([w.lower() for w in text if w not in stopwords])


# In[15]:


stopwords = set(nltk.corpus.stopwords.words('english'))

all_eths_text = {k:[] for k in all_eth_reviews.keys()}

for key, value in all_eth_reviews.items():
    print(key)
    for i in range(int(len(value))):
        if i % 10000 == 0:
            print("\r" + str(i), end="")
            
        doc = nltk.tokenize.word_tokenize(all_eth_reviews[key][i]['text'])
        
        text = [token for token in doc if re.match("[a-zA-Z0-9']+", token)]
        
        all_eths_text[key].append([w.lower() for w in text if w not in stopwords])
    print("")


# After removing proper nouns using Stanford POS tagger and removing punctuation we see:

# In[148]:


cleaned = [word.text for sent in doc.sentences for word in sent.words if word.upos != 'PROPN']
cleaned


# In[150]:


[token for token in cleaned if re.match("[a-zA-Z0-9']+", token)]


# Repeating this process for the whole dataset as well as removing stopwords,

# In[151]:


stopwords = set(nltk.corpus.stopwords.words('english'))

classifier_text = {k:[] for k in ternary_keys}

for key, value in ternary_reviews.items():
    print(key)
    for i in range(int(len(value))):
        if i % 1000 == 0:
            print(i)
            
        doc = nlp(ternary_reviews[key][i]['text'])
        
        cleaned = [word.text for sent in doc.sentences for word in sent.words if word.upos != 'PROPN']
        text = [token for token in cleaned if re.match("[a-zA-Z0-9']+", token)]
        
        classifier_text[key].append([w.lower() for w in text if w not in stopwords])


# Save our tokenized reviews as json for future use

# In[8]:


with open('nltk_tokenized_punc_removed_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(classifier_text, outfile, ensure_ascii=False, indent=4)


# In[34]:


with open('nltk_tokenized_punc_removed_all_eths_minus_american_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(all_eths_text, outfile, ensure_ascii=False, indent=4)


# In[16]:


with open('nltk_tokenized_punc_removed_all_eths_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(all_eths_text, outfile, ensure_ascii=False, indent=4)


# In[19]:


all_eths_text['Chinese'][5]


# ## 3. Topic Modeling

# Use the tomotopy library to perform LDA and MG-LDA on our dataset. Runs faster and easier to filter out words than gensim. Remove top 20 words and use 40 topics.

# In[ ]:


models = {}
numTopics = 40
numRemove = 20

for key in ternary_keys:
    models[key] = tp.LDAModel(k=numTopics, rm_top=numRemove)


# *Tuning area, to be deleted later*

# In[10]:


eth = 'Chinese'

for review in classifier_text[eth]:
    models[eth].add_doc(review)
    
for i in range(0, 100, 10):
    models[eth].train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, models[eth].ll_per_word))
    
# 40 topics

for k in range(models[eth].k):
    print('Top 10 words of topic #{}'.format(k))
    print(models[eth].get_topic_words(k, top_n=10))


# Observe above that sometimes we get punctuation in topics

# ## 4. Classifiers

# Create lexicons for bow representations of data using gensim

# In[3]:


classifier_text = {}

with open('nltk_tokenized_punc_removed_reviews.json', 'r') as infile:
    classifier_text =json.load(infile)


# In[ ]:


all_eths_text = {}

with open('nltk_tokenized_punc_removed_all_eths_minus_american_reviews.json', 'r') as infile:
    all_eths_text =json.load(infile)


# In[20]:


lexicons = {k:{} for k in ternary_keys}
big_lexicon = corpora.Dictionary({})

for k in lexicons.keys():
    lexicons[k] = corpora.Dictionary(classifier_text[k])
    
for lexicon in lexicons.values():
    big_lexicon.merge_with(lexicon)


# In[21]:


lexicons = {k:{} for k in all_eths_text.keys()}
big_lexicon = corpora.Dictionary({})

for k in lexicons.keys():
    lexicons[k] = corpora.Dictionary(all_eths_text[k])
    
for lexicon in lexicons.values():
    big_lexicon.merge_with(lexicon)


# In[36]:


big_lexicon = corpora.Dictionary.load("big_lexicon_n_20_removed_all_eths_may_2021")


# In[22]:


big_lexicon.filter_n_most_frequent(20)


# In[23]:


big_lexicon.save("big_lexicon_n_20_removed_all_eths_may_2021")


# In[8]:


big_lexicon.save("big_lexicon_n_20_removed")


# In[ ]:


big_lexicon.save("big_lexicon_n_20_removed_all_eths_minus_american")


# In[24]:


big_lexicon.save("big_lexicon")


# In[38]:


big_lexicon.filter_extremes(no_below=2, no_above=1.0, keep_n=None)


# In[39]:


len(list(big_lexicon.token2id.keys()))


# In[40]:


big_lexicon.save("big_lexicon_n_20_removed_all_eths_no_below_2")


# In[ ]:


big_bows = []

for k in all_eths_text.keys():
    for review in all_eths_text[k]:
        big_bows.append(big_lexicon.doc2bow(review))


# Save bows to json

# In[14]:


with open('big_bows_nltk_tokenized_punc_removed.json', 'w',encoding='utf-8') as outfile:
    json.dump(big_bows, outfile, ensure_ascii=False, indent=4)


# In[11]:


with open('big_bows_nltk_tokenized_punc_removed_n20_removed.json', 'w',encoding='utf-8') as outfile:
    json.dump(big_bows, outfile, ensure_ascii=False, indent=4)


# In[66]:


with open('big_bows_nltk_tokenized_punc_removed_n20_removed_all_eths.json', 'w',encoding='utf-8') as outfile:
    json.dump(big_bows, outfile, ensure_ascii=False, indent=4)


# Create data to feed into classifiers

# In[14]:


big_bows = []
with open('big_bows_nltk_tokenized_punc_removed.json', 'r') as file:
    big_bows = json.load(file)


# In[6]:


big_bows = []
with open('big_bows_nltk_tokenized_punc_removed_n20_removed.json', 'r') as file:
    big_bows = json.load(file)


# In[67]:


len(big_bows)


# In[ ]:


y_vals = np.array([[0]*len(classifier_text['Chinese']) + [1]*len(classifier_text['American (Traditional)']) + [2]*len(classifier_text['Asian Fusion'])
                    +[3]*len(classifier_text['Italian']) + [4]*len(classifier_text['Mexican']) + [5]*len(classifier_text['Japanese'])  
                    +[6]*len(classifier_text['Indian']) + [7]*len(classifier_text['French'])])

big_tf_array = matutils.corpus2csc(big_bows)


# In[4]:


y_vals = np.array([[0]*202819 + [1]* 806830 + [2]*111310 
                    +[3]* 415229 + [4]* 443937 + [5]*253150  
                    +[6]*84297 + [7]*97172])

big_tf_array = matutils.corpus2csc(big_bows)


# In[ ]:


vals = []

for i, value in enumerate(all_eths_text.values()):
    vals += ([i] * len(value))

y_vals = np.array(vals)

big_tf_array = matutils.corpus2csc(big_bows)


# In[22]:


import lda
import numpy as np


# In[45]:


sparse.save_npz('./big_tf_array_all_eths_minus_american.npz', big_tf_array)


# Ternary reviews data set up

# In[10]:


y_vals = np.array([[0]*202819 + [1]* 415229+ [2]*443937])

big_tf_array = matutils.corpus2csc(big_bows)


# In[5]:


y_vals = np.array([[0]*103655 + [1]*379265 + [2]*85153 + [3]*21398 + [4]*470416 + [5]*57018 + [6] *29919 + [7]*33074 + [8]*87440 + [9]*314753 +[10]*36932 +[11]*93532 +[12]*117824 +[13]*16093 +[14]*499055+[15]*47838 +[16]*160182 + [17]*73302 + [18]*26625 +[19]*16385+ [20]*108484 + [21]*117491 +[22]*19941 +[23]*26054])


# In[2]:


big_tf_array = sparse.load_npz('./big_tf_array_all_eths_minus_american.npz')


# Shuffle data using sklearn method

# In[6]:


big_tf_array, y_vals, locs = shuffle(big_tf_array.transpose(), y_vals.transpose(), locs)


# Split data using 70-15-15 split

# In[75]:


# why two indicies?

index_stop_first = int((.7*y_vals.shape[0]))
index_stop_second = int((.85*y_vals.shape[0]))


X_train = big_tf_array[:index_stop_first, :]
y_train = y_vals[:index_stop_first, :].ravel()
locs_train = locs[:index_stop_first] 
N_train = y_train.size


X_val = big_tf_array[index_stop_first:index_stop_second, :]
y_val = y_vals[index_stop_first:index_stop_second, :]
locs_val = locs[index_stop_first:index_stop_second] 
N_val = y_val.size

X_test = big_tf_array[index_stop_second:, :]
y_test = y_vals[index_stop_second:, :]
locs_test = locs[index_stop_second:]
N_test = y_test.size


# In[7]:


index_stop_first = int((.7*y_vals.shape[0]))
index_stop_second = int((.85*y_vals.shape[0]))


X_train = big_tf_array[:index_stop_first, :]
y_train = y_vals[:index_stop_first]
locs_train = locs[:index_stop_first] 
N_train = y_train.size


X_val = big_tf_array[index_stop_first:index_stop_second, :]
y_val = y_vals[index_stop_first:index_stop_second]
locs_val = locs[index_stop_first:index_stop_second] 
N_val = y_val.size

X_test = big_tf_array[index_stop_second:, :]
y_test = y_vals[index_stop_second:]
locs_test = locs[index_stop_second:]
N_test = y_test.size


# Save data for future use

# In[8]:


from scipy import sparse


# In[ ]:


sparse.save_npz('./data_n_20_removed_all_eths/X_train.npz', X_train)
#sparse.save_npz('./data/y_train.npz', y_train)
sparse.save_npz('./data_n_20_removed_all_eths/X_val.npz', X_val)
#sparse.save_npz('./data/y_val.npz', y_val)
sparse.save_npz('./data_n_20_removed_all_eths/X_test.npz', X_test)
#sparse.save_npz('./data/y_test.npz', y_test)


# In[14]:


sparse.save_npz('./data_n_20_removed_all_eths_less_than_15000/X_train.npz', X_train)
#sparse.save_npz('./data/y_train.npz', y_train)
sparse.save_npz('./data_n_20_removed_all_eths_less_than_15000/X_val.npz', X_val)
#sparse.save_npz('./data/y_val.npz', y_val)
sparse.save_npz('./data_n_20_removed_all_eths_less_than_15000/X_test.npz', X_test)
#sparse.save_npz('./data/y_test.npz', y_test)


# In[9]:


sparse.save_npz('./data_n_20_removed_all_eths_minus_american/X_train.npz', X_train)
#sparse.save_npz('./data/y_train.npz', y_train)
sparse.save_npz('./data_n_20_removed_all_eths_minus_american/X_val.npz', X_val)
#sparse.save_npz('./data/y_val.npz', y_val)
sparse.save_npz('./data_n_20_removed_all_eths_minus_american/X_test.npz', X_test)
#sparse.save_npz('./data/y_test.npz', y_t


# This stuff causes python to crash when reloading data

# In[15]:


np.save("./data_n_20_removed_all_eths_less_than_15000/y_train.npy", y_train)
#np.save("./data/X_val.npy", X_val)
np.save("./data_n_20_removed_all_eths_less_than_15000/y_val.npy", y_val)
#np.save("./data/X_test.npy", X_test)
np.save("./data_n_20_removed_all_eths_less_than_15000/y_test.npy", y_test)


# In[10]:


np.save("./data_n_20_removed_all_eths_minus_american/y_train.npy", y_train)
#np.save("./data/X_val.npy", X_val)
np.save("./data_n_20_removed_all_eths_minus_american/y_val.npy", y_val)
#np.save("./data/X_test.npy", X_test)
np.save("./data_n_20_removed_all_eths_minus_american/y_test.npy", y_test)


# In[23]:


np.load("./data/y_train.npy")


# In[15]:


with open('./data_n_20_removed_all_eths_minus_american/locs_train.json', 'w',encoding='utf-8') as outfile:
    json.dump(locs_train, outfile, ensure_ascii=False, indent=4)

with open('./data_n_20_removed_all_eths_minus_american/locs_val.json', 'w',encoding='utf-8') as outfile:
    json.dump(locs_val, outfile, ensure_ascii=False, indent=4)

with open('./data_n_20_removed_all_eths_minus_american/locs_test.json', 'w',encoding='utf-8') as outfile:
    json.dump(locs_test, outfile, ensure_ascii=False, indent=4)


