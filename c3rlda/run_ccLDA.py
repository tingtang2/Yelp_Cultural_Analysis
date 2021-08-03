import c3rlda
from scipy import sparse
import numpy as np

import argparse

if __name__ == '__main__':
    
    my_parser = argparse.ArgumentParser(description='Get number of topics for topic model')
    my_parser.add_argument('-num_topics', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    print(f"Topics: {args.num_topics}")
    
    
    num_topics = args.num_topics
    # get training data
    X_train = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_train.npz')
    cc= np.load("/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/y_train.npy")
        
    # filter for non zeros
    ix = X_train.getnnz(1)>0
    X_train = X_train[ix]
    cc= cc[ix]

    # run model 
    model = c3rlda.c3rLDA(n_topics=num_topics, n_iter=1000)
    model.fit(X_train.astype(np.intc), cc.astype(np.intc))

    # save big matricies
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzwc.npy", model.nzwc_)
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzw.npy", model.nzw_)
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/ndz.npy", model.ndz_)
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nx.npy", model.nx_)
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nz.npy", model.nz_)
    np.save(f"/scratch/tc2vg/ccLDA_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzc.npy", model.nzc_)
    
    # save space
    del model
    
    print(f"Done with {num_topics} topics")
