import c3rlda
from scipy import sparse
import numpy as np
import sklearn
from joblib import dump

import json
import gensim 
import argparse

#num_topics = [50, 100, 200]


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Get threshold number for min number of documents token must be present in')
    my_parser.add_argument('-below', type=int, dest='below', default=5, help = "threshold for min # of docs from preprocessing")
    my_parser.add_argument('-num_topics', type=int, dest='num_topics', default=25, help = "Number of topics for topic model")
    
    args = my_parser.parse_args()
    num_topics = args.num_topics
    print(f"Threshold: {args.below}")
    
    # get training data
    X_train = sparse.load_npz(f'/scratch/tc2vg/Yelp Project/data_below_{args.below}/X_train.npz')
    cc= np.load(f"/scratch/tc2vg/Yelp Project/data_below_{args.below}/y_train.npy")
    locs = []
    with open(f"/scratch/tc2vg/Yelp Project/data_below_{args.below}/locs_train.json", 'r') as file:
        locs = json.load(file)
        
    # filter for non zeros
    ix = X_train.getnnz(1)>0
    X_train = X_train[ix]
    cc= cc[ix]
    locs= np.array(locs)[ix]
    
    # do location run
    # locations of each ethnicity
    locs_eth = [[] for eth in range(len(set(cc)))]
    indicies_locs_eth = []

    for i, c in enumerate(cc):
        locs_eth[c].append(locs[i])
        indicies_locs_eth.append(len(locs_eth[c]) -1)
        
    n_regions = []

    eth_regions = [[] for eth in range(len(set(cc)))] # assignments of regions for each ethnicity

    for i in range(len(set(cc))):
        regions = 10
        converged = False

        while not converged: 
            dpgmm = sklearn.mixture.BayesianGaussianMixture(verbose=1, n_components=regions, max_iter=500)
            pred = dpgmm.fit_predict(locs_eth[i]).tolist()

            if all(w >= 0.005 for w in dpgmm.weights_):
                converged = True
                eth_regions[i] = pred
                n_regions.append(regions)
                print("Saving location model")
                dump(dpgmm, f"/scratch/tc2vg/loc_models/cuisine-{i}-below-{args.below}-dpgmm.joblib")
            else:
                regions += -1
    
    # get location assignment data from previous run   
    #reg_assign_mat = np.load("./model_persistence/all_eths_less_than_15000_1000_its_25_topics/reg_assign_mat.npy")
    #indicies_locs_eth = np.load("./model_persistence/all_eths_less_than_15000_1000_its_25_topics/indicies_locs_eth.npy")

    # 10 max regions
    #n_regions = [10]
    
    # do regional assigment set up
    reg_assign = []

    for i, elem in enumerate(eth_regions):
        reg_assign.append(np.array(elem, dtype=np.intc))

    max_len = max([x.shape[0] for x in reg_assign])

    reg_assign_mat = np.zeros((len(reg_assign), max_len), dtype = np.intc) # region assignment by cuisines and number on loc

    for i, row in enumerate(reg_assign):
        reg_assign_mat[i, :len(row)] += row

    indicies_locs_eth = np.array(indicies_locs_eth, dtype=np.intc) # placement in region index for each loc 
    
    # run model 
    model = c3rlda.c3rLDA(n_topics=num_topics, region_assignments=reg_assign_mat, n_regions_fluid = n_regions,
                    indices_region_assignments=indicies_locs_eth, n_iter=1000)
    model.fit_fluid(X_train.astype(np.intc), cc.astype(np.intc), locs)

    # save big matricies
    np.save(f"/scratch/tc2vg/c3rLDA_threshold_runs/all_eths_less_than_15000_1000_its_{args.below}_below/nzwcr.npy", model.nzwcr_)
    np.save(f"/scratch/tc2vg/c3rLDA_threshold_runs/all_eths_less_than_15000_1000_its_{args.below}_below/nzw.npy", model.nzw_)
    np.save(f"/scratch/tc2vg/c3rLDA_threshold_runs/all_eths_less_than_15000_1000_its_{args.below}_below/ndz.npy", model.ndz_)
    
    # save space
    del model
    
    print(f"Done with {args.below} below")
