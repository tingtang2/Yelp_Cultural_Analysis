from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump
from scipy import sparse

if __name__ == '__main__':
    X_train = sparse.load_npz('/scratch/tc2vg/Yelp Project/data_n_20_removed_all_eths_less_than_15000/X_train.npz')
    print("loaded training matrix")
    
    num_topics = 5
    print(f"Topics: {num_topics}")
    
    model = LatentDirichletAllocation(n_components = num_topics, verbose=1)
    print("Initialized model")
    
    print("Begin fitting")
    model.fit(X_train)
    
    print("Saving model")
    dump(model, f"/scratch/tc2vg/LDA_runs/{num-topics}-topics-lda.joblib")
