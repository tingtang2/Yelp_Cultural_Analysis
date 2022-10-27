# coding=utf-8
"""c3rLDA using collapsed Gibbs sampling"""

from __future__ import absolute_import, division, unicode_literals  # noqa
import logging
import sys

import numpy as np
from numpy import ma

import _c3rlda
import utils

import sklearn.mixture
from tqdm import tqdm

#logger = logging.getLogger('lda')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange

#################################

class c3rLDA:
    """c3rLDA using collapsed Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 2000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    beta : float, default 0.01
        Dirichlet parameter for distribution over words

    random_state : int or RandomState, optional
        The generator used for the initial topics.

    Attributes
    ----------
    `components_` : array, shape = [n_topics, n_features]
        Point estimate of the topic-word distributions (Phi in literature)
    `topic_word_` :
        Alias for `components_`
    `nzw_` : array, shape = [n_topics, n_features]
        Matrix of counts recording topic-word assignments in final iteration.
    `ndz_` : array, shape = [n_samples, n_topics]
        Matrix of counts recording document-topic assignments in final iteration.
    `doc_topic_` : array, shape = [n_samples, n_features]
        Point estimate of the document-topic distributions (Thbeta in literature)
    `nz_` : array, shape = [n_topics]
        Array of topic assignment counts in final iteration.

    """
    def __init__(self, n_topics, n_regions= 10, n_regions_fluid = [5, 7 ,6],
     region_assignments = None, indices_region_assignments = None, n_iter=1500, alpha=0.1, beta=0.01,
     delta =.01, gamma_0 = 1.0, gamma_1 = 1.0, Delta = .1, mu_0 = np.array([44, -103], dtype=np.double),
     S_0 = np.array([[1, 0], [0, 1]], dtype=np.double), lambda_0 = 1, v_0 =3, random_state=None, refresh=10):

        self.n_topics = n_topics
        self.n_regions = n_regions
        self.reg_assign = region_assignments
        self.ind_reg_assign = indices_region_assignments
        self.n_regions_fluid = n_regions_fluid
        self.n_iter = n_iter

        # params for basic LDA
        self.alpha = alpha        # param for Dirichlet prior for topic mixture
        self.beta = beta          # param for Dirichlet prior for dist of words for each topic
        self.delta= delta

        # params for ethnic cuisines
        self.gamma_0 = gamma_0    # first param for Beta prior for indicator
        self.gamma_1 = gamma_1    # second param for Beta prior for indicator

        self.Delta = Delta        # param for Dirichlet prior for dist of regions

        # params for location (Normal-Wishart conjugate prior)
        self.mu_0 = mu_0
        self.S_0 = S_0
        self.lambda_0 = lambda_0
        self.v_0 = v_0

        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh

        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be greater than zero")

        # random numbers that are reused
        rng = utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    def fit_fluid(self, X, cc, ls, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        cc: array of collections corresponding the ethnic cuisine type of each restaurant
        ls: array of locations coordinates for each review

        Returns
        -------
        self : object
            Returns the instance itself."""

        self._fit_fluid(X, cc, ls)
        return self


    def fit_complete(self, X, cc, ls, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        cc: array of collections corresponding the ethnic cuisine type of each restaurant
        ls: array of locations coordinates for each review

        Returns
        -------
        self : object
            Returns the instance itself."""

        self._fit_complete(X, cc, ls)
        return self


    def fit(self, X, cc, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        cc: array of collections corresponding the ethnic cuisine type of each restaurant

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, cc)
        return self



    def _fit_fluid(self, X, cc, ls):
        """
        Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """

        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize_fluid(X, cc, ls)

        with tqdm(total= self.n_iter) as pbar:
            for it in range(self.n_iter):
                # FIXME: using numpy.roll with a random shift might be faster
                random_state.shuffle(rands)
                self._sample_topics_fluid(rands)
                if it % self.refresh == 0:
                    #print(str(it) + "/" + str(self.n_iter)) #+ ": " + str(self.loglikelihood_complete()))
                    pbar.update(self.refresh)


        self.components_ = (self.nzw_ + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]



        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        del self.CS

        return self


    def _fit_complete(self, X, cc, ls):
        """
        Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """

        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize_complete(X, cc, ls)

        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            self._sample_topics_complete(rands)
            if it % self.refresh == 0:
                print(str(it) + "/" + str(self.n_iter) + ": " + str(self.loglikelihood_complete()))



        sum_ndz = self.ndz_
        sum_nzw = self.nzw_

        sum_nzwcr = self.nzwcr_



        # 15 samples, 1000 lag
        for it in range(self.n_iter):
            self._sample_topics_complete(rands)

            if it % 100 == 0:
                sum_ndz += self.ndz_
                sum_nzw += self.nzw_

                sum_nzwcr += self.nzwcr_
                print(str(it) + "/" + str(self.n_iter))


        self.ndz_ = sum_ndz / 15.0
        self.nzw_ = sum_nzw / 15.0

        self.nzwcr_ = sum_nzwcr / 15.0


        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # word distributions for individual collections
        self.topic_word_collection_ = (self.nzwcr_ + self.delta).astype(float)
        #self.topic_word_collection_ /= np.sum(self.topic_word_collection_, axis=1)[:, np.newaxis, :, :]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        del self.CS

        return self


    def _fit(self, X, cc):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X, cc)

        # Burn in
        with tqdm(total= self.n_iter) as pbar:
            for it in range(self.n_iter):
                # FIXME: using numpy.roll with a random shift might be faster
                random_state.shuffle(rands)
                self._sample_topics(rands)
                if it % self.refresh == 0:
                    #print(str(it) + "/" + str(self.n_iter))
                    pbar.update(self.refresh)


        sum_ndz = self.ndz_
        sum_nzw = self.nzw_

        sum_nzwc = self.nzwc_


        self.components_ = (self.nzw_ + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_

        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # word distributions for individual collections
        self.topic_word_collection_ = (self.nzwc_ + self.delta).astype(float)
        self.topic_word_collection_ /= np.sum(self.topic_word_collection_, axis=1)[:, np.newaxis, :]


        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        del self.XS
        del self.CS

        return self

    def _initialize_fluid(self, X, cc, ls):
        D, W = X.shape  # documents and vocab size
        N = int(X.sum()) # number of total tokens
        C = len(set(cc)) # number of collections
        n_topics = self.n_topics
        n_regions_fluid = self.n_regions_fluid
        n_iter = self.n_iter

        max_regions = np.max(n_regions_fluid)

        # for background distribution
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        # for ethnic cuisines x region distributions
        self.nzwcr_ =  np.zeros((n_topics, W, C, max_regions), dtype=np.intc) # phis for each region
        self.nzcr_ = np.zeros((n_topics, C, max_regions), dtype=np.intc) # topic counts for each region


        self.nx_ = nx_ = np.zeros((2, C, n_topics), dtype=np.intc) # indicators for each collection for each topic

        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.CS = CS = cc


        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)

        # TODO: check if initializing xs as zeros or random is better
        self.XS = XS = np.empty_like(self.WS, dtype=np.intc)


        self.LS = LS = ls

        np.testing.assert_equal(N, len(WS))

        for i in range(N):
            w, d, = WS[i], DS[i]
            c = CS[d]
            j = self.ind_reg_assign[d]

            rr = self.reg_assign[c][j]

            x_new = i % 2
            XS[i] = x_new

            z_new = i % n_topics
            ZS[i] = z_new

            ndz_[d, z_new] += 1
            nx_[x_new, c, z_new] += 1

            if x_new == 0:
                nzw_[z_new, w] += 1
                nz_[z_new] += 1
            else:
                self.nzwcr_[z_new, w, c, rr] += 1
                self.nzcr_[z_new, c, rr] += 1


        print("Finished initializing")
        self.loglikelihoods_ = []


    # just for ethnicities
    def _initialize(self, X, cc):
        D, W = X.shape  # documents and vocab size
        N = int(X.sum()) # number of total tokens
        C = len(set(cc)) # number of collections
        n_topics = self.n_topics
        #n_regions = self.n_regions
        n_iter = self.n_iter

        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_collections: {}".format(C))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        # for background distribution (normal LDA)
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        # for ethnic cuisines distributions
        self.nzwc_ = nzwc_ =  np.zeros((n_topics, W, C), dtype=np.intc) # phis for each collection
        self.nzc_ = nzc_ = np.zeros((n_topics, C), dtype=np.intc) # topic counts for each collection
        self.nx_ = nx_ = np.zeros((2, C, n_topics), dtype=np.intc) # indicators for each collection for each topic


        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X) # word and document indices
        self.CS = CS = cc # ethnic cuisine for each review

        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc) # topics for each word

        # TO DO: check if initializing xs as zeros or random is better
        self.XS = XS = np.random.binomial(np.ones(self.WS.shape[0], dtype=np.intc), .5) # indicator for background
        #self.XS = XS = np.zeros(self.WS.shape[0], dtype=np.intc) # indicator for background
        XS = XS.astype('intc')
        self.XS = XS

        np.testing.assert_equal(N, len(WS))
        np.testing.assert_equal(len(set(DS)), len(CS))


        for i in range(N):
            w, d, x = WS[i], DS[i], XS[i]
            c = CS[d]

            z_new = i % n_topics
            ZS[i] = z_new

            ndz_[d, z_new] += 1
            nx_[x, c, z_new] += 1

            if x == 0:
                nzw_[z_new, w] += 1
                nz_[z_new] += 1
            else:
                nzwc_[z_new, w, c] += 1
                nzc_[z_new, c] += 1

        print("Finished initializing")
        self.loglikelihoods_ = []

    def _initialize_complete(self, X, cc, ls):
        D, W = X.shape  # documents and vocab size
        N = int(X.sum()) # number of total tokens
        C = len(set(cc)) # number of collections
        n_topics = self.n_topics
        n_regions = self.n_regions
        n_iter = self.n_iter

        # for background distribution
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)


        eth_reg_data = [] # allow for region fluidity

        # for ethnic cuisines x region distributions
        self.nzwcr_ = nzwcr_ =  np.zeros((n_topics, W, C, n_regions), dtype=np.intc) # phis for each collection x region
        self.nzcr_ = nzcr_ = np.zeros((n_topics, C, n_regions), dtype=np.intc) # topic counts for each collection x region

        self.nx_ = nx_ = np.zeros((2, C, n_regions, n_topics), dtype=np.intc) # indicators for each collection for each topic

        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.CS = CS = cc

        dpgmm = sklearn.mixture.BayesianGaussianMixture(verbose=1, n_components=n_regions, max_iter=500)

        self.RS = RS = dpgmm.fit_predict(ls) # regions for each doc
        self.RS = RS = RS.astype('intc')
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)

        # TODO: check if initializing xs as zeros or random is better
        self.XS = XS = np.empty_like(self.WS, dtype=np.intc)


        self.LS = LS = ls

        np.testing.assert_equal(N, len(WS))

        for i in range(N):
            w, d, = WS[i], DS[i]
            c = CS[d]
            rr = RS[d]

            x_new = i % 2
            XS[i] = x_new

            z_new = i % n_topics
            ZS[i] = z_new

            ndz_[d, z_new] += 1
            nx_[x_new, c,rr, z_new] += 1

            if x_new == 0:
                nzw_[z_new, w] += 1
                nz_[z_new] += 1
            elif x_new ==1:
                nzwcr_[z_new, w, c, rr] += 1
                nzcr_[z_new, c, rr] += 1


        print("Finished initializing")
        self.loglikelihoods_ = []


    def _initializeLs(self, ls):
            n_topics = self.n_topics
            n_regions = self.n_regions
            n_iter = self.n_iter

            self.D = D = len(ls)

            self.ndr_ = ndr_ = np.zeros((D, n_regions), dtype=np.intc) # sigmas for each doc
            self.nr_ = nr_ = np.zeros(n_regions, dtype=np.intc)
            #self.nrl_ = nrl_ = np.zeros((n_regions, D), dtype=np.single) # regions by location


            self.RS = RS = np.zeros(D, dtype=np.intc)# regions for each doc

            # TO DO: check if initializing xs as zeros or random is better
            #self.XS = XS = np.random.binomial(np.ones(self.WS.shape[0], dtype=np.intc), .5) # indicator for background
            self.LS = LS = ls


            for i in range(len(ls)):
                r_new = i % n_regions
                RS[i] = r_new

                ndr_[i, r_new] += 1
                nr_[r_new] += 1


            print("Finished initializing")
            self.loglikelihoods_ = []


    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        beta = self.beta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return _lda._loglikelihood(nzw, ndz, nz, nd, alpha, beta)

    def loglikelihood_complete(self):
        """Calculate complete log likelihood, log p(w,z,x)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """

        self.components_ = (self.nzw_ + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # word distributions for individual collections
        self.topic_word_collection_ = (self.nzwcr_ + self.delta).astype(float)
        self.topic_word_collection_ /= np.sum(self.topic_word_collection_, axis=1)[:, np.newaxis, :]


        # indicator distributions
        self.indicator_word_collection_region_ = (self.nx_ + self.gamma_0)
        self.indicator_word_collection_region_ /= np.sum(self.indicator_word_collection_region_, axis=3)[:, :, :, np.newaxis]

        likelihood = 0

        prevDoc = self.DS[0]
        docLikelihood = 0
        for i in range(self.ndz_.shape[0]):
            w = self.WS[i]
            d = self.DS[i]
            z = self.ZS[i]
            c = self.CS[d]
            x = self.XS[i]
            rr = self.RS[d]


            if prevDoc == d:
                if x == 0:
                    docLikelihood += np.log(self.doc_topic_[d, z] * self.topic_word_[z, w])
                else:
                    docLikelihood += np.log(self.doc_topic_[d, z] * self.topic_word_collection_[z, w, c, rr])
            else:
                likelihood += docLikelihood
                docLikelihood = 0
                prevDoc = d

        return likelihood


    def _sample_topics(self, rands):
        """Samples all topic assignments. Called once per iteration.

        Calls Cython routine for speed
        """

        n_topics, vocab_size = self.nzw_.shape

        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        beta = np.repeat(self.beta, vocab_size).astype(np.float64)
        delta = np.repeat(self.delta, vocab_size).astype(np.float64) # for cross collection

        _lda._sample_topics(self.WS, self.DS, self.ZS, self.CS, self.XS,
         self.nx_, self.nzw_, self.ndz_, self.nz_, self.nzwc_, self.nzc_, alpha,
         beta, delta, self.gamma_0, self.gamma_1, rands)


    def _sample_topics_complete(self, rands):
        """Samples all topic assignments. Called once per iteration.

        Calls Cython routine for speed
        """

        n_topics, vocab_size = self.nzw_.shape

        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        beta = np.repeat(self.beta, vocab_size).astype(np.float64)
        delta = np.repeat(self.delta, vocab_size).astype(np.float64) # for cross collection
        gamma = np.repeat(self.gamma_0, 3).astype(np.float64)

        _lda._sample_topics_complete(self.WS, self.DS, self.ZS, self.CS, self.XS, self.RS, self.LS,
         self.nx_, self.nzw_, self.ndz_, self.nz_, self.nzwcr_, self.nzcr_, alpha, beta, delta, gamma, rands)

    def _sample_topics_fluid(self, rands):
        """Samples all topic assignments. Called once per iteration.

        Calls Cython routine for speed
        """

        n_topics, vocab_size = self.nzw_.shape

        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        beta = np.repeat(self.beta, vocab_size).astype(np.float64)
        delta = np.repeat(self.delta, vocab_size).astype(np.float64) # for cross collection


        _lda._sample_topics_fluid(self.WS, self.DS, self.ZS, self.CS, self.XS, self.reg_assign, self.ind_reg_assign, self.nzwcr_,
         self.nzcr_, self.nx_, self.nzw_, self.ndz_, self.nz_, alpha, beta, delta, self.gamma_0, rands)
