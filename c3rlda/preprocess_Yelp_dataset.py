#!/usr/bin/env python
# coding: utf-8

# Ting Chen

import json

import re

import numpy as np
import nltk

from gensim import corpora
from sklearn.utils import shuffle

from scipy import sparse


# ## 1. Load in data


reviews = []
for line in open('yelp_dataset/yelp_academic_dataset_review.json', 'r'):
    reviews.append(json.loads(line))



businesses = []
for line in open('yelp_dataset/yelp_academic_dataset_business.json', 'r'):
    businesses.append(json.loads(line))


## collect ids and tags

business_ids = []
business_tags = set()
tag_counter = Counter()

for b in businesses:
    for tag in re.split(',', str(b['categories'])):
        if tag[0] is ' ':
            tag = tag[1:]
        if "Food" in str(b['categories']) or "Restaurants" in str(b['categories']) or "Bars" in str(b['categories']):
            business_tags.add(tag)
            tag_counter.update([tag])
            business_ids.append(b['business_id'])


## compile reviews of restaurants 
restaurant_reviews = []

for i in range(len(reviews)):
    if reviews[i]['business_id'] in business_ids:
        restaurant_reviews.append(reviews[i])
    if i % 10000 == 0:
        print(i)


with open('restaurant_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(restaurant_reviews, outfile, ensure_ascii=False, indent=4)


############################################################################################3


all_eths = []

with open('all_eths.txt', 'r') as infile:
    all_eths = infile.read().split('\n')


## get stuff from json dump

#restaurant_reviews = []

#with open('restaurant_reviews.json', 'r') as file:
#    restaurant_reviews = json.load(file)




## begin looking at different cuisines
ethnic_ids = {k:[] for k in all_eths}

for b in businesses:
    for tag in re.split(',', str(b['categories'])):
        if tag[0] is ' ':
            tag = tag[1:]
        for k in ethnic_ids.keys():
            if tag == k:
                ethnic_ids[k].append(b['business_id'])


for k, v in ethnic_ids.items():
    print(k, len(v))


del ethnic_ids['']


ethnic_reviews = {k:[] for k in all_eths}

for i in range(len(restaurant_reviews)):
    if i % 10000 == 0:
        print(i)
    r = restaurant_reviews[i]
    for k in ethnic_ids.keys():
        if r['business_id'] in ethnic_ids[k]:
            ethnic_reviews[k].append(r)



with open('all_eths_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(ethnic_reviews, outfile, ensure_ascii=False, indent=4)


#####################################################################################


#all_eth_reviews = {}

#with open('all_eths_reviews.json', 'r') as file:
#    all_eth_reviews = json.load(file)


# down sample
too_small = []
for key, value in all_eth_reviews.items():
    if len(value) < 15000:
        too_small.append(key)


for key in too_small:
    del all_eth_reviews[key]



del all_eth_reviews['Canadian (New)']
del all_eth_reviews['Ethnic Food']
del all_eth_reviews['Asian Fusion']
del all_eth_reviews['Halal']
del all_eth_reviews['New Mexican Cuisine']
#del all_eth_reviews['American (New)']
#del all_eth_reviews['American (Traditional)']
del all_eth_reviews['Irish']
del all_eth_reviews['Cantonese']

for key, value in all_eth_reviews.items():
    print(key + str(':'), len(value))

#with open('all_eths_minus_american_keys.json', 'w',encoding='utf-8') as outfile:
#    json.dump(list(all_eth_reviews.keys()), outfile, ensure_ascii=False, indent=4)


# Get locations for each review


businesses = []
for line in open('yelp_dataset/yelp_academic_dataset_business.json', 'r'):
    businesses.append(json.loads(line))



bus_table = {}

for bus in businesses:
    bus_table[bus['business_id']] = (bus['latitude'], bus['longitude'])



locations = []
for key, value in all_eth_reviews.items():
    for review in value:
        bus_id = review['business_id']
        locations.append(bus_table[bus_id])

locs = np.array(locations)


with open('locs_all_eths_less_than_15000.json', 'w',encoding='utf-8') as outfile:
    json.dump(locations, outfile, ensure_ascii=False, indent=4)

#locs = []

#with open('locs_all_eths_minus_american.json', 'r') as infile:
#    locs = json.load(infile)


# ## 2. Preprocess


# For now best results are nltk tokenized and punctation removed

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



with open('nltk_tokenized_punc_removed_all_eths_reviews.json', 'w',encoding='utf-8') as outfile:
    json.dump(all_eths_text, outfile, ensure_ascii=False, indent=4)



# Observe above that sometimes we get punctuation in topics

lexicons = {k:{} for k in all_eths_text.keys()}
big_lexicon = corpora.Dictionary({})

for k in lexicons.keys():
    lexicons[k] = corpora.Dictionary(all_eths_text[k])
    
for lexicon in lexicons.values():
    big_lexicon.merge_with(lexicon)



#big_lexicon = corpora.Dictionary.load("big_lexicon_n_20_removed_all_eths_may_2021")


big_lexicon.filter_n_most_frequent(20)


#big_lexicon.save("big_lexicon_n_20_removed_all_eths_may_2021")


big_lexicon.filter_extremes(no_below=2, no_above=1.0, keep_n=None)
big_lexicon.save("big_lexicon_n_20_removed_all_eths_no_below_2")

