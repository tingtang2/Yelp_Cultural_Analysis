## c3rLDA - Cross Collection Cross Regional Latent Dirichlet Allocation

a topic modeling package written for our paper: 

Key utilities written in Cython for simple but fast computations

Code is adapted and heavily inspired by the lda PyPi library as well as Michael Paul's ccLDA implementation

### Installation

To Cythonize:
```bash

python3 setup.py --inplace

```

### Preprocessing the data

```bash

python3 preprocess_Yelp_dataset.py

```

### Running the topic model
```bash

python3 run.py

```

### Running the coherence metrics
