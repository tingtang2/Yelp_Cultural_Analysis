import numpy as np
from tqdm import tqdm
from palmettopy.palmetto import Palmetto

palmetto = Palmetto()

# For standard base topics
def get_base_metrics(words):
    coherence_measures = ['npmi', 'ca', 'cp', 'cv', 'umass']

    for m in coherence_measures:
        topic_coherence = []

        for i in tqdm(range(words.shape[0])):
            topic_coherence.append(palmetto.get_coherence(words=list(words[i, :].astype(str)), coherence_type=m))

        print(m, np.mean(topic_coherence))
        
        
# For cuisine specific topics
def get_cuisine_metrics(words_cuisine, num_cuisines=28):
    coherence_measures = ['npmi', 'ca', 'cp', 'cv', 'umass']

    for m in coherence_measures:
        topic_coherence = []
    
        for c in tqdm(range(num_cuisines)):
            for i in range(words_cuisine.shape[0]):
                topic_coherence.append(palmetto.get_coherence(words=list(words_cuisine[i, :, c].astype(str)), coherence_type=m))
    
    print(m, np.mean(topic_coherence))
