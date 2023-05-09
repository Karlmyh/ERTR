# Author: dtrckd <dtrck@gmail.com>  from sklearn/lineear_model/cd_fast.pyx
# License: BSD 3 clause

from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings

#ctypedef np.float64_t DOUBLE
#ctypedef np.uint32_t UINT32_t
#ctypedef floating (*DOT)(int N, floating *X, int incX, floating *Y,
#                         int incY) nogil
#ctypedef void (*AXPY)(int N, floating alpha, floating *X, int incX,
#                      floating *Y, int incY) nogil
#ctypedef floating (*ASUM)(int N, floating *X, int incX) nogil

np.import_array()

# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef floating diff_abs_max(int n, floating* a, floating* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY) nogil
    void saxpy "cblas_saxpy"(int N, float alpha, float *X, int incX,
                             float *Y, int incY) nogil
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil
    float sdot "cblas_sdot"(int N, float *X, int incX, float *Y,
                            int incY) nogil
    double dasum "cblas_dasum"(int N, double *X, int incX) nogil
    float sasum "cblas_sasum"(int N, float *X, int incX) nogil
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                           double *X, int incX, double *Y, int incY,
                           double *A, int lda) nogil
    void sger "cblas_sger"(CBLAS_ORDER Order, int M, int N, float alpha,
                           float *X, int incX, float *Y, int incY,
                           float *A, int lda) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             int M, int N, double alpha, double *A, int lda,
                             double *X, int incX, double beta,
                             double *Y, int incY) nogil
    void sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             int M, int N, float alpha, float *A, int lda,
                             float *X, int incX, float beta,
                             float *Y, int incY) nogil
    double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil
    float snrm2 "cblas_snrm2"(int N, float *X, int incX) nogil
    void dcopy "cblas_dcopy"(int N, double *X, int incX, double *Y,
                             int incY) nogil
    void scopy "cblas_scopy"(int N, float *X, int incX, float *Y,
                            int incY) nogil
    void dscal "cblas_dscal"(int N, double alpha, double *X, int incX) nogil
    void sscal "cblas_sscal"(int N, float alpha, float *X, int incX) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
#cpdef double cydot(floating[::1] x, floating[::1] y, int dim):
cpdef double cydot(double[:] x, double[:] y, int dim):

    ## fused types version of BLAS functions
    #cdef DOT dot
    #cdef AXPY axpy
    #cdef ASUM asum

    #if floating is float:
    #    dtype = np.float32
    #    dot = sdot
    #    axpy = saxpy
    #    asum = sasum
    #else:
    #    dtype = np.float64
    #    dot = ddot
    #    axpy = daxpy
    #    asum = dasum
    ##dot = ddot

    ## get the data information into easy vars
    #cdef unsigned int n_samples = X.shape[0]
    #cdef unsigned int n_features = X.shape[1]

    # Get the pointers.
    #cdef floating* x_ptr = &x[0]
    #cdef floating* y_ptr = &y[0]
    cdef double* x_ptr = &x[0]
    cdef double* y_ptr = &y[0]

    #return dot(dim, x_ptr, 1, y_ptr, 1)
    return ddot(dim, x_ptr, 1, y_ptr, 1)



