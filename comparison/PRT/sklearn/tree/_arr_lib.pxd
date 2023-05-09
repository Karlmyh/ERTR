
from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t
ctypedef floating (*DOT)(int N, floating *X, int incX, floating *Y,
                         int incY) nogil
ctypedef void (*AXPY)(int N, floating alpha, floating *X, int incX,
                      floating *Y, int incY) nogil
ctypedef floating (*ASUM)(int N, floating *X, int incX) nogil

#cpdef double cydot(floating[::1] x, floating[::1] y, int dim)
cpdef double cydot(double[:] x, double[:] y, int dim)
