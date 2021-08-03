import numpy as np
import argparse


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Get number of topics for topic model')
    my_parser.add_argument('-k', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    print(f"Topics: {args.num_topics}")
    
    num_topics = args.num_topics
    c3rLDA_nzwcr_ = np.load(f"/scratch/tc2vg/c3rLDA_model_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzwcr.npy")
    
    
    reg_assign_mat = np.load("/home/tc2vg/Yelp_Analysis_Code/lda/model_persistence/all_eths_less_than_15000_1000_its_25_topics/reg_assign_mat.npy")
    n_regions = np.max(reg_assign_mat, axis=1) # number of regions assigned to each cuisine
    
    print("data loaded")

    # Marginalize region out of nzwcr
    nzwc = np.zeros(c3rLDA_nzwcr_.shape[:-1])

    for i in range(nzwc.shape[0]):
        print('\rTopic: ', i, end="")
        for j in range(nzwc.shape[1]):
            for k in range(nzwc.shape[2]):
                nzwc[i, j, k] = np.sum(c3rLDA_nzwcr_[i, j, k, :n_regions[k]])
    
    np.save(f"/scratch/tc2vg/c3rLDA_model_runs/all_eths_less_than_15000_1000_its_{num_topics}_topics/nzwc.npy", nzwc)