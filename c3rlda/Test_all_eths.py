import lda
from scipy import sparse
import numpy as np

import json
import gensim 
import sklearn
import sys

from scipy.spatial.distance import jensenshannon

# load data

X_train = sparse.load_npz('/Users/tingchen/Desktop/Yelp Project/data_n_20_removed_all_eths/X_train.npz')
cc= np.load("/Users/tingchen/Desktop/Yelp Project/data_n_20_removed_all_eths/y_train.npy")
locs = []
with open("/Users/tingchen/Desktop/Yelp Project/data_n_20_removed_all_eths/locs_train.json", 'r') as file:
    locs = json.load(file)

ix = X_train.getnnz(1)>0
X_train = X_train[ix]
cc= cc[ix]
locs= np.array(locs)[ix]

# locations of each ethnicity
locs_eth = [[] for eth in range(len(set(cc)))]
indicies_locs_eth = []

for i, c in enumerate(cc):
    locs_eth[c].append(locs[i])
    indicies_locs_eth.append(len(locs_eth[c]) -1)

# determine regions

n_regions = []

eth_regions = [[] for eth in range(len(set(cc)))] # assignments of regions for each ethnicity

for i in range(len(set(cc))):
    regions = 10
    converged = False

    while not converged:
        dpgmm = sklearn.mixture.BayesianGaussianMixture(verbose=1, n_components=regions, max_iter=500)
        pred = dpgmm.fit_predict(locs_eth[i]).tolist()

        if all(i >= 0.005 for i in dpgmm.weights_):
            converged = True
            eth_regions[i] = pred
            n_regions.append(regions)
        else:
            regions += -1

# train model

reg_assign = []

for i, elem in enumerate(eth_regions):
    reg_assign.append(np.array(elem, dtype=np.intc))

max_len = max([x.shape[0] for x in reg_assign])
reg_assign_mat = np.zeros((len(reg_assign), max_len), dtype = np.intc)

for i, row in enumerate(reg_assign):
    reg_assign_mat[i, :len(row)] += row

indicies_locs_eth = np.array(indicies_locs_eth, dtype=np.intc)

first = 60000

model = lda.LDA(n_topics= 25, region_assignments=reg_assign_mat[:first], n_regions_fluid = n_regions, indices_region_assignments=indicies_locs_eth, n_iter=20)
model.fit_fluid(X_train.astype(np.int)[:first], cc.astype(np.int32)[:first], locs[:first])

# Test Jensen Shannon 

print("Try adding one")
model.nzwcr_ += 1 
