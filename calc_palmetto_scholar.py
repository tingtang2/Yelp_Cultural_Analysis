import numpy as np
import argparse
import coherence_calc_methods as coh
import json
from tqdm import tqdm

from palmettopy.palmetto import Palmetto

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Get number of topics for topic model')
    my_parser.add_argument('-k', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    print(f"Topics: {args.num_topics}")
    
    num_topics = args.num_topics
    
    # load and sort stuff
    scholar_nzwc = np.load(f"/scratch/tc2vg/scholar_runs/{num_topics}_topics/beta_ci.npz")['beta']
    
    top10_cuisine = np.argsort(scholar_nzwc, axis=1)[:, -10:]
    
    vocab = []
    with open('/scratch/tc2vg/scholar_data/train.vocab.json', 'r', encoding='utf-8') as input_file:
        for line in input_file:
            vocab= json.loads(line, encoding='utf-8')
            
    words = np.array(vocab)[top10_cuisine.astype(np.intc)]
    
    # computational stuff
    coherence_measures = ['npmi', 'ca', 'cp', 'cv', 'umass']
    output = {}
    palmetto = Palmetto()
    exceptions = 0

    for m in coherence_measures:
        topic_coherence = []

        for i in tqdm(range(words.shape[0])):
            try:
                topic_coherence.append(palmetto.get_coherence(words=list(words[i, :].astype(str)), coherence_type=m))
            except:
                exceptions += 1
                print(f"exceptions = {exceptions}")

        #print(m, np.mean(topic_coherence))
        output[m] = np.mean(topic_coherence)
    
    with open(f'./coherence_results/palmetto-scholar-{num_topics}.json', 'w') as outfile:
        json.dump(output, outfile)