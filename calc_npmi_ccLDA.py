import numpy as np
import argparse
from tqdm import tqdm
from scipy import sparse
import json

X_val = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_val.npz')
X_test = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_test.npz')
cc_val= np.load("/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/y_val.npy")
cc_test= np.load("/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/y_test.npy")

total_c = np.concatenate((cc_val, cc_test))
total_X = sparse.vstack([X_val, X_test])

ix = total_X.getnnz(1)>0
total_X = total_X[ix]
total_c= total_c[ix]

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

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Get number of topics for topic model')
    my_parser.add_argument('-k', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    print(f"Topics: {args.num_topics}")
    
    num_topics = args.num_topics
    
    ccLDA_nzwc = np.load(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzwc.npy")
    
    top10_cuisine = np.argsort(ccLDA_nzwc, axis=1)[:, -10:, :]
    
    cclda_npmi = []
    output = {}

    for c in tqdm(range(ccLDA_nzwc.shape[2])):
        cclda_npmi.append(calculate_npmi(top10_cuisine[:, :, c]))
    
    output['npmi-internal'] = np.mean(cclda_npmi)
    
    with open(f'./coherence_results/internal-npmi-ccLDA-{num_topics}.json', 'w') as outfile:
        json.dump(output, outfile)
