#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
from scipy.special import gamma

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

from libc.stdio cimport printf

from libc.math cimport exp, pow, sqrt
from libc.math cimport abs as cabs

M_PI = 3.14159265358979323846

from   scipy.special import gammaln


def pdf(x, mean, shape, df):
    return 1000000 * np.exp(logpdf(x, mean, shape, df))


def logpdf(x, mean, shape, df):
    p = x.shape[1] if len(x.shape) == 2 else x.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + p)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = p/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E

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


"""

def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:] CS, int[:] XS, int[:] RS, double[:, :] LS, int[:, :, :, :] nx,int[:, :] nzw,
                  int[:, :] ndz, int[:] nz, int[:, :, :, :] nzwcr, int[:, :, :] nzcr, int[:, :] ndr, int[:] nr,
                  double[:] Delta, double[:] alpha, double[:] beta, double[:] delta,
                  double gamma_0, double gamma_1, double[:] rands, double lambda_0, double[:, :] S_0, double[:] mu_0,
                  double v_0):
    cdef int i, k, w, d, c, z, z_new, x, x_new, rr, r_new
    cdef double r
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef int n_regions = nr.shape[0]

    cdef double dist_cum = 0
    cdef double dist_cum_x = 0
    cdef double dist_cum_r = 0

    cdef double beta_sum = 0
    cdef double delta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    cdef double* dist_sum_x = <double*> malloc(2 * sizeof(double))


    C = np.zeros((2, 2), dtype=np.double)

    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")


    y_bar = np.mean(LS)

    inv_RS = {i:[] for i in range(n_regions)}

    for i in range(len(RS)):
        inv_RS[RS[i]].append(i)


    #print("\n" + str(inv_RS))

    # gibbs sampling for regions
    for d in range(len(RS)):

        rr = RS[d]
        print("d", d, "rr", rr)
        ndr[d, rr] -= 1

        dist_sum_r = np.zeros(n_regions)

        for reg in range(n_regions):
            #print("reg", reg)
            N_ = nr[reg]

            Is = inv_RS[reg]
            #print(Is)

            # find constant C
            for i in Is:
                L = np.array(LS[i], dtype=(np.double, np.double))
                #p
                printfrint(L)
                #print(((L - y_bar).T).dot((L - y_bar)))
                C += ((L - y_bar).T).dot((L - y_bar))

            #print("C", C)

            S_N = np.array(S_0) + np.array(C) + (lambda_0 * N_/(lambda_0+ N_)) * (y_bar - mu_0).dot((y_bar - mu_0).T)
            S_N1 = np.array(S_0) + np.array(C) + (lambda_0 * N/(lambda_0+ N_-1)) * (y_bar - mu_0).dot((y_bar - mu_0).T)

            prob = pow(M_PI, -2/2)*pow((lambda_0+N)/(lambda_0+N_-1), -2/2)*pow(np.linalg.det(S_N), -(v_0+N_)/2)/pow(np.linalg.det(S_N1), -(v_0+N_-1)/2)*exp(lgamma((v_0 + N_)/2))/exp(lgamma((v_0 + N_ -2)/2))

            #print("computed prob")

            dist_cum_r += (ndr[d, reg] + Delta[reg]) * prob
            dist_sum_r[reg] = dist_cum_r


        #r = rands[d % n_rand] * dist_cum_r

        #print("index for rand, d:", d % n_rand)

        rr_new = np.searchsorted(dist_sum_r, np.random.rand() * dist_cum_r, side='left')
        #rr_new = np.searchsorted(dist_sum_r, r, side='right')
        #rr_new = searchsorted(dist_sum, n_regions, np.random.rand() * dist_cum_r)
        #printf("z_new %d\n", z_new)

        #print("rr_new", rr_new)

        if rr_new == -1:
            rr_new = rr_new +1
            print("oh no")

        #inc(ndr[d, rr_new])
        ndr[d, rr_new] += 1
        #print("increment count")
        RS[d] = rr_new
        #print("assign")

    #free(dist_sum_r)

    with nogil:

        # first equation
        printf("here1 \n")
        for i in range(beta.shape[0]):
            beta_sum += beta[i]

        # second equation for when x = 1
        for i in range(delta.shape[0]):
            delta_sum += delta[i]

        # gibbs sampling for topics and indicator variables
        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]
            c = CS[d]
            x = XS[i]
            rr = RS[d]

            dec(nx[x, c, z, rr])
            dec(ndz[d, z])

            if x == 0:
                dec(nzw[z, w])
                dec(nz[z])
            else:
                dec(nzwcr[z, w, c, rr])
                dec(nzcr[z, c, rr])

            dist_cum_x = (nx[0,c,z, rr] + gamma_0) * (ndr[d, rr] + Delta[rr]) * (nzw[z,w] + beta[w]) / (nz[z] + beta_sum)
            dist_sum_x[0] = dist_cum_x

            dist_cum_x += (nx[1,c,z,rr] + gamma_1) * (ndr[d, rr] + Delta[rr]) * (nzwcr[z,w,c, rr] + delta[w]) / (nzcr[z,c, rr] + delta_sum)
            dist_sum_x[1] = dist_cum_x


            r = rands[i % n_rand] * dist_cum_x

            #printf("%d\nindex for rand, i:", i % n_rand)

            #x_new = searchsorted(dist_sum_x, 2, r)

            if r < dist_sum_x[0]:
                x_new = 0
            else:
                x_new = 1


            if x_new == 0:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += ((nzw[k, w] + beta[w]) / (nz[k] + beta_sum) * (ndz[d, k] + alpha[k])) * (ndr[d, rr] + Delta[rr])
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)
            else:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += (nzwcr[k, w, c, rr] + delta[w]) / (nzcr[k, c, rr] + delta_sum) * (ndz[d, k] + alpha[k]) * (ndr[d, rr] + Delta[rr])
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)
                #printf("z_new %d\n", z_new)


            if x_new == 0:
                inc(nzw[z_new, w])
                inc(nz[z_new])
            else:
                inc(nzwcr[z_new, w, c, rr])
                inc(nzcr[z_new, c, rr])

            inc(nx[x, c, z_new, rr])
            inc(ndz[d, z_new])

            ZS[i] = z_new
            XS[i] = x_new


        free(dist_sum)
        free(dist_sum_x)

"""
'''
def _update_covariance(L, x, n):
    if n == 0:
        return
    cholupdate(L, x)
    _update_covariance(L, x, n-1)


cdef inline FLOAT_t hypot(FLOAT_t x,FLOAT_t y):
    cdef FLOAT_t t
    x = cabs(x)
    y = cabs(y)
    t = x if x < y else y
    x = x if x > y else y
    t = t/x
    return x*sqrt(1+t*t)

cdef cholupdate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x):
    """
    Update the upper triangular Cholesky factor R with the rank 1 addition
    implied by x such that:
    R_'R_ = R'R + outer(x,x)
    where R_ is the upper triangular Cholesky factor R after updating.  Note
    that both x and R are modified in place.
    """
    cdef unsigned int p
    cdef unsigned int k
    cdef unsigned int i
    cdef FLOAT_t r
    cdef FLOAT_t c
    cdef FLOAT_t s
    cdef FLOAT_t a
    cdef FLOAT_t b

    p = <unsigned int>len(x)
    for k in range(p):
        r = hypot(R[<unsigned int>k,<unsigned int>k], x[<unsigned int>k])
        c = r / R[<unsigned int>k,<unsigned int>k]
        s = x[<unsigned int>k] / R[<unsigned int>k,<unsigned int>k]
        R[<unsigned int>k,<unsigned int>k] = r
        #TODO: Use BLAS instead of inner for loop
        for i in range(<unsigned int>(k+1),<unsigned int>p):
            R[<unsigned int>k,<unsigned int>i] = (R[<unsigned int>k,<unsigned int>i] + s*x[<unsigned int>i]) / c
            x[<unsigned int>i] = c * x[<unsigned int>i] - s * R[<unsigned int>k,<unsigned int>i]

'''
def _sample_Ls(int[:] RS, double[:, :] LS, int[:, :] ndr, int[:] nr, double[:] Delta, double[:] rands, 
        double lambda_0, double[:, :] S_0, double[:] mu_0, double v_0):
    
    cdef int i, k, w, d, c, z, z_new, x, x_new, rr, r_new
    cdef double r
    cdef int n_rand = rands.shape[0]
    cdef int n_regions = nr.shape[0]

    cdef double dist_cum_r

    cdef double beta_sum = 0
    cdef double delta_sum = 0


    # dict for region to doc index mappings
    inv_RS = {i:[] for i in range(n_regions)}

    for i in range(len(RS)):
        inv_RS[RS[i]].append(i)

    #print("\n" + str(inv_RS))
    
    S_0 = np.array(S_0)

    # cholesky decomposition of prior parameter
    #L_0 = np.linalg.cholesky(S_0, lower=False)

    # gibbs sampling for regions
    for d in range(len(RS)):
        rr = RS[d]
        ll = LS[d]
        
        ndr[d, rr] -= 1

        dist_sum_r = np.zeros(n_regions)

        dist_cum_r = 0
        for reg in range(n_regions):
            #print("reg", reg)
            N_ = nr[reg]

            # doc indices for region
            Is = inv_RS[reg]
            
            # update parameters of NIW

            # sum of square differences compared to mean
            C = np.zeros((2, 2), dtype=np.double)
            
            y_bar = np.mean([LS[i] for i in Is])

            for i in Is:
                L = np.array(LS[i], dtype=(np.double, np.double))
                #print(((L - y_bar).T).dot((L - y_bar)))
                C += ((L - y_bar).T).dot((L - y_bar))

            #print("C", C)

            lambda_n = lambda_0 + N_

            v_n = v_0 + N_

            mu_n1 = (lambda_0 * np.array(mu_0) + ((N_ - 1) * y_bar))/(lambda_n -1)

            df = v_n -1 -2 +1
            
            S_N = S_0 + C + (lambda_0 * N_/(lambda_0+ N_)) * (y_bar - mu_0).dot((y_bar - mu_0).T)
            S_N1 = S_0 + C + (lambda_0 * N_/(lambda_0+ N_-1)) * (y_bar - mu_0).dot((y_bar - mu_0).T)

            '''

            prob = pow(M_PI, -2/2)*pow((lambda_0+N_)/(lambda_0+N_-1), -2/2)*pow(np.linalg.det(S_N), -(v_0+N_)/2)/pow(np.linalg.det(S_N1), -(v_0+N_-1)/2)*exp(lgamma((v_0 + N_)/2))/exp(lgamma((v_0 + N_ -2)/2)) '''

            #_update_covariance(L_0, , N_)

            var = S_N1 * (lambda_n)/((lambda_n -1) * (v_n -1 - 2 + 1))

            det = np.linalg.det(var)

            inv = np.linalg.inv(var)

            #print("exponent", -(df +2)/2)

            prob = (((ll - mu_n1).T.dot(inv)).dot(ll - mu_n1)) # ** (-(df +2)/2) #((1 + (1/df)* ((ll - mu_n1).T.dot(inv)).dot(ll - mu_n1)) ** (-(df +2)/2) )
 #gamma((df + 2)/2)/(gamma(df/2)* M_PI * (det ** 0.5)) * 

            #prob = -1 * logpdf(ll, mu_n1, var, df)
            
            #print("computed prob", prob)

            dist_cum_r += (ndr[d, reg] + Delta[reg]) * prob
            dist_sum_r[reg] = dist_cum_r


        r = rands[d % n_rand] * dist_cum_r

        #print("index for rand, d:", d % n_rand)

        rr_new = np.searchsorted(dist_sum_r, np.random.rand() * dist_cum_r, side='left')
        #rr_new = np.searchsorted(dist_sum_r, r, side='right')
        #rr_new = searchsorted(dist_sum, n_regions, np.random.rand() * dist_cum_r)
        #printf("z_new %d\n", z_new)

        print("d", d, "rr_new", rr_new)

        #inc(ndr[d, rr_new])
        ndr[d, rr_new] += 1
        
        RS[d] = rr_new

    #free(dist_sum_r)

def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:] CS, int[:] XS, int[:, :, :] nx, int[:, :] nzw,
                  int[:, :] ndz, int[:] nz, int[:, :, :] nzwc, int[:, :] nzc,  double[:] alpha, double[:] beta, double[:] delta,
                  double gamma_0, double gamma_1, double[:] rands):
    cdef int i, k, w, d, c, z, z_new, x, x_new
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]

    cdef double r 

    cdef double dist_cum, dist_cum_x

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

        # gibbs sampling for topics and indicator variables
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

            #printf("%d\nindex for rand, i:", i % n_rand)

            #x_new = searchsorted(dist_sum_x, 2, r)

            if r < dist_sum_x[0]:
                x_new = 0
            else:
                x_new = 1

            dist_cum = 0
            if x_new == 0:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += ((nzw[k, w] + beta[w]) / (nz[k] + beta_sum) * (ndz[d, k] + alpha[k]))    
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)
               # printf("z_new, background %d\n", z_new)
            else:
                for k in range(n_topics):
                    # beta is a double so cdivision yields a double
                    dist_cum += (nzwc[k, w, c] + delta[w]) / (nzc[k, c] + delta_sum) * (ndz[d, k] + alpha[k])  
                    dist_sum[k] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                z_new = searchsorted(dist_sum, n_topics, r)
               # printf("z_new %d\n", z_new)


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


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double beta, double Delta) nogil:
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

        '''# calculate log p(r)
        for d in range(D):
            ll += (lgamma(Delta * n_regions) -
                    lgamma(Delta * n_regions + nr[d]))
            for r in range(n_regions):
                if ndr[d, r] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha'''
        return ll
