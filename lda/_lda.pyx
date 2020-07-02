#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

from libc.stdio cimport printf

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive: " + str(x))
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:] CS, int[:] XS, int[:, :, :] nx,int[:, :] nzw,
                  int[:, :] ndz, int[:] nz, int[:, :, :] nzwc, int[:, :] nzc, double[:] alpha, double[:] beta, double[:] delta,
                  double gamma_0, double gamma_1, double[:] rands):
    cdef int i, k, w, d, c, z, z_new, x, x_new
    cdef double r
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double dist_cum = 0
    cdef double dist_cum_x = 0
    cdef double beta_sum = 0
    cdef double delta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    cdef double* dist_sum_x = <double*> malloc(2 * sizeof(double))

    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:

        # first equation
        for i in range(beta.shape[0]):
            beta_sum += beta[i]

        # second equation for when x = 1
        for i in range(delta.shape[0]):
            delta_sum += delta[i]

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]
            c = CS[d]
            x = XS[i]

            dec(nx[x, c, z])
            dec(ndz[d, z])

            if x == 0:
                dec(nzw[z, w])
                dec(nz[z])
            else:
                dec(nzwc[z, w, c])
                dec(nzc[z, c])

            dist_cum_x = (nx[0,c,z] + gamma_0) * (nzw[z,w] + beta[w]) / (nz[z] + beta_sum)
            dist_sum_x[0] = dist_cum_x

            dist_cum_x += (nx[1,c,z] + gamma_1) * (nzwc[z,w,c] + delta[w]) / (nzc[z,c] + delta_sum)
            dist_sum_x[1] = dist_cum_x


            r = rands[i % n_rand] * dist_cum_x

            #x_new = searchsorted(dist_sum_x, 2, r)

            if r < dist_sum_x[0]:
                x_new = 0
            else:
                x_new = 1

            printf("x_new %d\n", x_new)

            if x_new == 0:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += (nzw[k, w] + beta[w]) / (nz[k] + beta_sum) * (ndz[d, k] + alpha[k])
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)
            else:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += (nzwc[k, w, c] + delta[w]) / (nzc[k, c] + delta_sum) * (ndz[d, k] + alpha[k])
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)


            if x_new == 0:
                inc(nzw[z_new, w])
                inc(nz[z_new])
            else:
                inc(nzwc[z_new, w, c])
                inc(nzc[z_new, c])

            inc(nx[x, c, z_new])
            inc(ndz[d, z_new])

            ZS[i] = z_new
            XS[i] = x_new


        free(dist_sum)
        free(dist_sum_x)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double beta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_beta, lgamma_alpha
    with nogil:
        lgamma_beta = lgamma(beta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(beta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(beta * vocab_size + nz[k])

            printf("nz[k]: \n")

            printf("%d\n", nz[k])

            ## DEBUG
            #printf("%d\n", nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(beta + nzw[k, w]) - lgamma_beta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll
