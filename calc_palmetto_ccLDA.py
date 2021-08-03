import numpy as np
import argparse
import coherence_calc_methods as coh
import json
from tqdm import tqdm
from gensim.corpora import Dictionary
from palmettopy.palmetto import Palmetto



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Get number of topics for topic model')
    my_parser.add_argument('-k', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    print(f"Topics: {args.num_topics}")
    
    num_topics = args.num_topics
    
    # load and sort stuff
    ccLDA_nzwc = np.load(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzwc.npy")
    
    top10_cuisine = np.argsort(ccLDA_nzwc, axis=1)[:, -10:, :]
    
    our_lexicon = Dictionary.load("big_lexicon_n_20_removed_all_eths_less_than_15000")

    inv_lexicon= {v: k for k, v in our_lexicon.token2id.items()}

    words = np.vectorize(inv_lexicon.get)(top10_cuisine)
    # computational stuff
    coherence_measures = ['npmi', 'ca', 'cp', 'cv', 'umass']
    output = {}
    palmetto = Palmetto()
    exceptions = 0

    for m in coherence_measures:
        topic_coherence = []
    
        for c in tqdm(range(28)):
            for i in range(words.shape[0]):
                try: 
                    topic_coherence.append(palmetto.get_coherence(words=list(words[i, :, c].astype(str)), coherence_type=m))
                except:
                    exceptions += 1
                    print(f"exceptions = {exceptions}")
        
        output[m] = np.mean(topic_coherence)
    
    with open(f'./coherence_results/palmetto-ccLDA-{num_topics}.json', 'w') as outfile:
        json.dump(output, outfile)
