## c3rLDA - Cross Collection Cross Regional Latent Dirichlet Allocation

a topic modeling package written for our paper: Analyzing Cultural Assimilation through the Lens of Yelp Restaurant Reviews

Key utilities written in Cython for simple but fast computations

Code is adapted and heavily inspired by the lda PyPi library as well as Michael Paul's ccLDA implementation

### Installation

To Cythonize:
```bash

python3 setup.py build_ext -b c3rlda/

```

### Preprocessing the data

Please download the dataset from [the Yelp Open Dataset website](https://www.yelp.com/dataset "Yelp Open Dataset").

Of those files, we use `yelp_academic_dataset_review.json` and `yelp_academic_dataset_business.json`.

To filter the dataset: 
```bash

python3 preprocess_Yelp_dataset.py

```

To tokenize: 
```bash

python3 preprocess_data.py -below 5

```

### Running the topic model
```bash

python3 run_{model}.py

```

### Running the coherence metrics
```bash

python3 calc_NPMI_internal.py

```

```bash

python3 calc_palmetto_{model}.py

```
