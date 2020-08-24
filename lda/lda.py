# coding=utf-8
"""Latent Dirichlet allocation using collapsed Gibbs sampling"""

from __future__ import absolute_import, division, unicode_literals  # noqa
import logging
import sys

import numpy as np

import _lda
import utils

logger = logging.getLogger('lda')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange

#################################

class LDA:
    """Latent Dirichlet allocation using collapsed Gibbs sampling

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

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> import lda
    >>> model = lda.LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.

    Wallach, Hanna M., Iain Murray, Ruslan Salakhutdinov, and David Mimno. 2009.
    “Evaluation Methods for Topic Models.” In Proceedings of the 26th Annual
    International Conference on Machine Learning, 1105–1112. ICML ’09. New York,
    NY, USA: ACM. https://doi.org/10.1145/1553374.1553515.

    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.

    """
    def __init__(self, n_topics, n_regions= 10, n_iter=1000, alpha=0.1, beta=0.01, delta =.01, gamma_0 = 1.0, gamma_1 = 1.0, Delta = .1, mu_0 = np.array([44, -103], dtype=np.double), S_0 = np.array([[1, 0], [0, 1]], dtype=np.double), lambda_0 = 1, v_0 =3, random_state=None, refresh=10):
        self.n_topics = n_topics
        self.n_regions = n_regions
        self.n_iter = n_iter

        '''def __init__(self, n_topics, n_iter=1000, alpha=0.1, beta=0.01, delta =.01, gamma_0 = 1.0, gamma_1 = 1.0, Delta = .1, random_state=None, refresh=10):'''

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


    def fit_complete(self, X, cc, ls, y=None):
        Fit the model with X.

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
            Returns the instance itself.

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

    def fitLs(self, ls):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        ls: array of locations

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fitLs(ls)
        return self

    def fit_transform(self, X, y=None):
        """Apply dimensionality reduction on X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        self._fit(X)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        To calculate an approximation of the distribution over topics for each
        new document this function uses the "iterated pseudo-counts" approach
        described in Wallach, Murray, Salakhutdinov, and Mimno (2009) and
        justified in greater dbetail in Buntine (2009). Specifically, we
        implement the "simpler first order version" described in section 3.3 of
        Buntine (2009).

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = utils.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        thbeta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(thbeta_doc) == self.n_topics
        assert thbeta_doc.shape == (self.n_topics,)
        return thbeta_doc

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
            '''
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            '''
            self._sample_topics_complete(rands)
            if it % self.refresh == 0:
                print(str(it) + "/" + str(self.n_iter))


        #ll = self.loglikelihood()
        #logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        del self.XS
        del self.CS
        del self.RS

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
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            '''
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            '''
            self._sample_topics(rands)
            if it % self.refresh == 0:
                print(str(it) + "/" + str(self.n_iter))


        #ll = self.loglikelihood()
        #logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
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

    def _fitLs(self, Ls):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initializeLs(Ls)
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            '''
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            '''
            self._sample_Ls(rands)
            if it % self.refresh == 0:
                print(str(it) + "/" + str(self.n_iter))


        #ll = self.loglikelihood()
        #logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.LS
        del self.RS

        return self

    def _initialize_complete(self, X, cc, ls):
        D, W = X.shape  # documents and vocab size
        N = int(X.sum()) # number of total tokens
        C = len(set(cc)) # number of collections
        n_topics = self.n_topics
        n_regions = self.n_regions
        n_iter = self.n_iter
        #logger.info("n_documents: {}".format(D))
        #logger.info("vocab_size: {}".format(W))
        #logger.info("n_words: {}".format(N))
        #logger.info("n_collections: {}".format(C))
        #logger.info("n_topics: {}".format(n_topics))
        #logger.info("n_iter: {}".format(n_iter))

        # for background distribution
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        # for ethnic cuisines x region distributions
        self.nzwc_ = nzwc_ =  np.zeros((n_topics, W, C), dtype=np.intc) # phis for each collection
        self.nzwr_ = nzwr_ =  np.zeros((n_topics, W, n_regions), dtype=np.intc) # phis for each region
        self.nzc_ = nzc_ = np.zeros((n_topics, C), dtype=np.intc) # topic counts for each collection
        self.nzr_ = nzr_ = np.zeros((n_topics, n_regions), dtype=np.intc) # topic counts for each collection

        self.nxc_ = nxc_ = np.zeros((2, C, n_topics), dtype=np.intc) # indicators for each collection for each topic
        self.nxr_ = nxr_ = np.zeros((2, n_regions, n_topics), dtype=np.intc) # indicators for each region for each topic


        self.ndr_ = ndr_ = np.zeros((D, n_regions), dtype=np.intc) # sigmas for each doc
        self.nr_ = nr_ = np.zeros(n_regions, dtype=np.intc)
        #self.nrl_ = nrl_ = np.zeros((n_regions, D), dtype=np.single) # regions by location


        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        # for regions
        self.CS = CS = cc
        self.RS = RS = np.empty_like(self.CS, dtype=np.intc) # regions for each doc
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)

        # TO DO: check if initializing xs as zeros or random is better
        #self.XS = XS = np.random.binomial(np.ones(self.WS.shape[0], dtype=np.intc), .5) # indicator for background
        self.XS = XS = np.zeros(self.WS.shape[0], dtype=np.intc) # indicator for background
        XS = XS.astype('intc')
        self.XS = XS

        self.LS = LS = ls

        np.testing.assert_equal(N, len(WS))

        for i in range(D):
            r_new = i % n_regions
            RS[i] = r_new

            ndr_[i, r_new] += 1
            nr_[r_new] += 1

        for i in range(N):
            w, d, x = WS[i], DS[i], XS[i]
            c = CS[d]
            rr = RS[d]

            z_new = i % n_topics
            ZS[i] = z_new


            ndz_[d, z_new] += 1
            nx_[x, c,rr, z_new] += 1

            if x == 0:
                nzw_[z_new, w] += 1
                nz_[z_new] += 1
            else:
                nzwcr_[z_new, w, c, r_new] += 1
                nzc_[z_new, c] += 1

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

    """def _sample_topics(self, rands):
        Samples all topic assignments. Called once per iteration.

        Calls Cython routine for speed


        n_topics, vocab_size = self.nzw_.shape

        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        beta = np.repeat(self.beta, vocab_size).astype(np.float64)
        delta = np.repeat(self.delta, vocab_size).astype(np.float64) # for cross collection
        Delta = np.repeat(self.Delta, self.n_regions).astype(np.float64) # for region

        _lda._sample_topics(self.WS, self.DS, self.ZS, self.CS, self.XS, self.RS, self.LS,
         self.nx_, self.nzw_, self.ndz_, self.nz_, self.nzwcr_, self.nzcr_, self.ndr_,
         self.nr_, Delta, alpha, beta, delta, self.gamma_0, self.gamma_1, rands, self.lambda_0,
         self.S_0, self.mu_0, self.v_0)
    """

    def _sample_Ls(self, rands):
        """
        Samples all topic assignments. Called once per iteration.

        Calls Cython routine for speed
        """

        Delta = np.repeat(self.Delta, self.n_regions).astype(np.float64) # for region

        _lda._sample_Ls(self.RS, self.LS, self.ndr_, self.nr_, Delta, rands,
        self.lambda_0, self.S_0, self.mu_0, self.v_0)

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

        _lda._sample_topics(self.WS, self.DS, self.ZS, self.CS, self.XS,
         self.nx_, self.nzw_, self.ndz_, self.nz_, self.nzwc_, self.nzc_, alpha,
         beta, delta, self.gamma_0, self.gamma_1, rands)
