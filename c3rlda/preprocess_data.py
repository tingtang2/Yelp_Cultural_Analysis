import numpy as np
from gensim.corpora import Dictionary
from gensim import matutils
import argparse
import lda
import json

from sklearn.utils import shuffle
from scipy import sparse

if __name__ == '__main__':
    
    my_parser = argparse.ArgumentParser(description='Get threshold number for min number of documents token must be present in')
    my_parser.add_argument('-below', type=int, dest='below', default=5, help = "threshold for min # of docs from preprocessing")
    
    args = my_parser.parse_args()
    print(f"Threshold: {args.below}")
    
    # load in data
    big_lexicon = Dictionary.load(f"/scratch/tc2vg/Yelp_Data/lexicons/big_lexicon_n_20_removed_all_eths_no_below_{args.below}")
    
    all_eths_text = {}

    with open('/scratch/tc2vg/Yelp_Data/nltk_tokenized_punc_removed_all_eths_reviews.json', 'r') as infile:
        all_eths_text =json.load(infile)
        
    locs = []

    with open('/scratch/tc2vg/Yelp_Data/locs_all_eths_less_than_15000.json', 'r') as infile:
        locs = json.load(infile)
        
    print("loaded data")
    # convert to bow
    
    big_bows = []

    for k in all_eths_text.keys():
        for review in all_eths_text[k]:
            big_bows.append(big_lexicon.doc2bow(review))
    
    print("converted to bow")
    
    # save bow
    with open(f'/scratch/tc2vg/Yelp_Data/bow_reps/big_bows_no_below_{args.below}.json', 'w',encoding='utf-8') as outfile:
        json.dump(big_bows, outfile, ensure_ascii=False, indent=4)
    
    # shuffle data
    vals = []

    for i, value in enumerate(all_eths_text.values()):
        vals += ([i] * len(value))

    y_vals = np.array(vals)

    big_tf_array = matutils.corpus2csc(big_bows)

    big_tf_array, y_vals, locs = shuffle(big_tf_array.transpose(), y_vals.transpose(), locs)
    
    # split into 70-15-15
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
    
    # save stuff in correct folder
    print("saving stuff")
    
    sparse.save_npz(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/X_train.npz', X_train)
    sparse.save_npz(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/X_val.npz', X_val)
    sparse.save_npz(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/X_test.npz', X_test)
    
    np.save(f"/scratch/tc2vg/Yelp Project/data_below_{args.below}/y_train.npy", y_train)
    np.save(f"/scratch/tc2vg/Yelp Project/data_below_{args.below}/y_val.npy", y_val)
    np.save(f"/scratch/tc2vg/Yelp Project/data_below_{args.below}/y_test.npy", y_test)
    
    with open(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/locs_train.json', 'w',encoding='utf-8') as outfile:
        json.dump(locs_train, outfile, ensure_ascii=False, indent=4)

    with open(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/locs_val.json', 'w',encoding='utf-8') as outfile:
        json.dump(locs_val, outfile, ensure_ascii=False, indent=4)

    with open(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/locs_test.json', 'w',encoding='utf-8') as outfile:
        json.dump(locs_test, outfile, ensure_ascii=False, indent=4)
    