# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from libc.stdio cimport printf

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.time cimport time,time_t

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport WeightedMedianCalculator

#
# _arr_lib import
#
#from ._arr_lib cimport *

import scipy.stats as stats
from scipy.optimize import linprog
from cpython cimport Py_INCREF, PyObject
cdef double INFINITY = np.inf

#from statsmodels.regression.quantile_regression import QuantReg

def prob_region(double x, double left_f, double right_f,  DOUBLE_t sigma):

    if left_f == -INFINITY:
        return stats.norm.cdf(right_f, x, sigma)
        # return stats.laplace.cdf(right_f, x, sigma)
        # return stats.t.cdf(right_f, 3, x, sigma)
        # return stats.t.cdf(right_f, 5, x, sigma)
        # return stats.gamma.cdf(right_f, 3, x, sigma)
        # return stats.gamma.cdf(right_f, 5, x, sigma)
        # return stats.lognorm.cdf(right_f, sigma, x, sigma)
    elif right_f == INFINITY:
        return 1 - stats.norm.cdf(left_f, x, sigma)
        # return 1 - stats.laplace.cdf(left_f, x, sigma)
        # return 1 - stats.t.cdf(left_f, 3, x, sigma)
        # return 1 - stats.t.cdf(left_f, 5, x, sigma)
        # return 1 - stats.gamma.cdf(left_f, 3, x, sigma)
        # return 1 - stats.gamma.cdf(left_f, 5, x, sigma)
        # return 1 - stats.lognorm.cdf(left_f, sigma, x, sigma)
    else:
        return stats.norm.cdf(right_f, x, sigma) - stats.norm.cdf(left_f, x, sigma)
        # return stats.laplace.cdf(right_f, x, sigma) - stats.laplace.cdf(left_f, x, sigma)
        # return stats.t.cdf(right_f, 3, x, sigma) - stats.t.cdf(left_f, 3, x, sigma)
        # return stats.t.cdf(right_f, 5, x, sigma) - stats.t.cdf(left_f, 5, x, sigma)
        # return stats.gamma.cdf(right_f, 3, x, sigma) - stats.gamma.cdf(left_f, 3, x, sigma)
        # return stats.gamma.cdf(right_f, 5, x, sigma) - stats.gamma.cdf(left_f, 5, x, sigma)
        # return stats.lognorm.cdf(right_f, sigma, x, sigma) - stats.lognorm.cdf(left_f, sigma, x, sigma)


cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        y_stride : SIZE_t
            y_stride is used to index the kth output value as follows:
            y[i, k] = y[i * y_stride + k]
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : DOUBLE_t
            The total weight of the samples being considered
        samples : array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node

        """

        pass

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double proxy_impurity_improvement2(self) nogil:
        return 0

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split

        Return
        ------
        double : improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right / 
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left / 
                             self.weighted_n_node_samples * impurity_left)))


    cdef int extra_init(self, object X, DOUBLE_t* sigmas, DOUBLE_t alpha):
        pass


    cdef int update2(self, SIZE_t new_pos, SIZE_t feature, DOUBLE_t threshold) nogil except -1:
        pass

    cdef int update3(self, SIZE_t new_pos, SIZE_t feature, DOUBLE_t threshold) nogil except -1:
        pass

    cdef int _set_region(self, SIZE_t region, Coord* path) nogil except -1:
        pass

    cdef int reset2(self) nogil except -1:
        pass

    cdef int set_preg(self, np.ndarray[DOUBLE_t, ndim=2] preg):
        pass

    cdef np.ndarray[DOUBLE_t, ndim=2] get_preg(self):
        pass

    cdef inline feat_bound(self, Coord* path, SIZE_t feature):
        ''' Return the uni-dimensional region length for the given feature. '''

        if path[0].is_root:
            return -INFINITY, INFINITY

        cdef bint is_left
        cdef DOUBLE_t thr_a
        cdef DOUBLE_t thr_b

        cdef int i = 0
        cdef bint first = True

        while True:

            while path[i].feature != feature:
                if path[i].is_end:
                    break
                i += 1

            if path[i].feature != feature and first == True:
                return -INFINITY, INFINITY
            elif first:
                first = False
                is_left = path[i].is_left
                thr_a = path[i].threshold

                if is_left:
                    thr_b = -INFINITY
                else:
                    thr_b = INFINITY

                if path[i].is_end:
                    break

                i += 1
                continue

            if path[i].feature == feature:
                if is_left:
                    if path[i].threshold < thr_a and path[i].threshold > thr_b:
                        thr_b = path[i].threshold
                else:
                    if path[i].threshold > thr_a and path[i].threshold < thr_b:
                        thr_b = path[i].threshold

            if path[i].is_end:
                break
            else:
                i += 1

        if is_left:
            #printf("(%f, %f)\n", thr_b, thr_a)
            return thr_b, thr_a
        else:
            #printf("(%f, %f)\n", thr_a, thr_b)
            return thr_a, thr_b


    cdef np.ndarray _get_y_ndarray(self):
        """Wraps value as a 1-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.n_samples
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(DOUBLE_t)
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.y)
        #arr = PyArray_NewFromDescr(np.ndarray, np.float64, 1, shape,
        #                           strides, <void*> self.y,
        #                           np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        y_stride : SIZE_t
            The stride between elements in the buffer, important if there
            are multiple targets (multi-output)
        sample_weight : array-like, dtype=DTYPE_t
            The weight of each sample
        weighted_n_samples : SIZE_t
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """

        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """

        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride


cdef class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

            sum_total += self.sum_stride

        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(ClassificationCriterion):
    r"""Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion."""


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : DTYPE_t
            The memory address to save the impurity of the left node to
        impurity_right : DTYPE_t
            The memory address to save the impurity of the right node to
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0


    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] += w * y_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] -= w * y_ik

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * self.y_stride + k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

cdef class MAE(RegressionCriterion):
    r"""Mean absolute error impurity criterion

       MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
       value and f_i is the predicted value."""
    def __dealloc__(self):
        """Destructor."""
        free(self.node_medians)

    cdef np.ndarray left_child
    cdef np.ndarray right_child
    cdef DOUBLE_t* node_medians

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.node_medians = NULL

        # Allocate memory for the accumulators
        safe_realloc(&self.node_medians, n_outputs)

        self.left_child = np.empty(n_outputs, dtype='object')
        self.right_child = np.empty(n_outputs, dtype='object')
        # initialize WeightedMedianCalculators
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""

        cdef SIZE_t i, p, k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w = 1.0

        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef void** left_child
        cdef void** right_child

        left_child = <void**> self.left_child.data
        right_child = <void**> self.right_child.data

        for k in range(self.n_outputs):
            (<WeightedMedianCalculator> left_child[k]).reset()
            (<WeightedMedianCalculator> right_child[k]).reset()

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]

                # push method ends up calling safe_realloc, hence `except -1`
                # push all values to the right side,
                # since pos = start initially anyway
                (<WeightedMedianCalculator> right_child[k]).push(y_ik, w)

            self.weighted_n_node_samples += w
        # calculate the node medians
        for k in range(self.n_outputs):
            self.node_medians[k] = (<WeightedMedianCalculator> right_child[k]).get_median()

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef SIZE_t i, k
        cdef DOUBLE_t value
        cdef DOUBLE_t weight

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        # reset the WeightedMedianCalculators, left should have no
        # elements and right should have all elements.

        for k in range(self.n_outputs):
            # if left has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> left_child[k]).size()):
                # remove everything from left and put it into right
                (<WeightedMedianCalculator> left_child[k]).pop(&value,
                                                               &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> right_child[k]).push(value,
                                                                 weight)
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        cdef DOUBLE_t value
        cdef DOUBLE_t weight
        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        # reverse reset the WeightedMedianCalculators, right should have no
        # elements and left should have all elements.
        for k in range(self.n_outputs):
            # if right has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> right_child[k]).size()):
                # remove everything from right and put it into left
                (<WeightedMedianCalculator> right_child[k]).pop(&value,
                                                                &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> left_child[k]).push(value,
                                                                weight)
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        # Update statistics up to new_pos
        #
        # We are going to update right_child and left_child
        # from the direction that require the least amount of
        # computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    # remove y_ik and its weight w from right and add to left
                    (<WeightedMedianCalculator> right_child[k]).remove(y_ik, w)
                    # push method ends up calling safe_realloc, hence except -1
                    (<WeightedMedianCalculator> left_child[k]).push(y_ik, w)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    # remove y_ik and its weight w from left and add to right
                    (<WeightedMedianCalculator> left_child[k]).remove(y_ik, w)
                    (<WeightedMedianCalculator> right_child[k]).push(y_ik, w)

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.pos = new_pos
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Computes the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        for k in range(self.n_outputs):
            dest[k] = <double> self.node_medians[k]

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]"""

        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t i, p, k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik

        cdef double impurity = 0.0

        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = samples[p]

                y_ik = y[i * self.y_stride + k]

                impurity += <double> fabs((<double> y_ik) - <double> self.node_medians[k])
        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).
        """

        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef SIZE_t i, p, k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t median

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        impurity_left[0] = 0.0
        impurity_right[0] = 0.0

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> left_child[k]).get_median()
            for p in range(start, pos):
                i = samples[p]

                y_ik = y[i * self.y_stride + k]

                impurity_left[0] += <double>fabs((<double> y_ik) -
                                                 <double> median)
        impurity_left[0] /= <double>((self.weighted_n_left) * self.n_outputs)

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> right_child[k]).get_median()
            for p in range(pos, end):
                i = samples[p]

                y_ik = y[i * self.y_stride + k]

                impurity_right[0] += <double>fabs((<double> y_ik) -
                                                  <double> median)
        impurity_right[0] /= <double>((self.weighted_n_right) *
                                      self.n_outputs)


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))







#
# 
# Warning : Weighted input not managed here....
#
cdef class MSEPROB(RegressionCriterion):
    """ Only works with bestSplitter2.
       
        NEW: New criterion for data with uncertainty

        MSE2 = (1 / n)*(\sum_i (y_i - F_i)**2), where y_i is the true
        value and f_i is the predicted value:
        F_i = P_i(P'P)**{-1}P' * y
        such as P is a matrix with each probability to be in each region
    """


    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

        self.fn = 0
        self.region = 0
        self.sigmas = NULL

        self.preg = np.zeros((self.n_samples,1), dtype=np.float64)
        self.preg_back_r = np.zeros(self.n_samples, dtype=np.float64)


    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:

        RegressionCriterion.init(self, y, y_stride, sample_weight, weighted_n_samples, samples,
                                 start, end)


        # Warning: n_output>1 not supported
        #printf('init criterion. n_samples: %d, n_outputs: %d\n', self.n_samples, self.n_outputs)

        self.reset()

        return 0

    cdef int reset(self) nogil except -1:
        RegressionCriterion.reset(self)


    cdef int reset2(self) nogil except -1:

        with gil:
            #self.gmma = self.gmma_back.copy()
            self.preg[:, self.region] = self.preg_back_r
            self.preg = np.delete(self.preg, self.preg.shape[1]-1, axis=1)

    # Called juste after init()
    cdef int _set_region(self, SIZE_t region, Coord* path) nogil except -1:

        self.region = region

        cdef int f

        with gil:
            self.preg_back_r = np.zeros(self.n_samples, dtype=np.float64)

            self.preg_back_r[:] = self.preg[:, region]

            self.region_bounds = np.zeros((self.n_features, 2), dtype=np.float64)

            for f in range(self.n_features):
                left_f, right_f = self.feat_bound(path, f)
                self.region_bounds[f, 0] = left_f
                self.region_bounds[f, 1] = right_f

    cdef int set_preg(self, np.ndarray[DOUBLE_t, ndim=2] preg):

        self.preg = preg
        return 0

    cdef np.ndarray[DOUBLE_t, ndim=2] get_preg(self):
        return np.asarray(self.preg)

    cdef int extra_init(self, object X, DOUBLE_t* sigmas, DOUBLE_t alpha):

        # Initialize X
        cdef np.ndarray X_ndarray = X
        #cdef SIZE_t n_samples = X.shape[0] # already in.

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        self.n_features = X.shape[1]


        self.sigmas = sigmas
        self._alpha = alpha


    cdef int update2(self, SIZE_t new_pos, SIZE_t feature, DOUBLE_t threshold) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef SIZE_t n_regions
        cdef double[:] left_region 
        cdef double[:] right_region
        cdef double[:] y

        cdef int f, i
        cdef double _pb_r, _pb_l, pb_r, pb_l, sigma, left_f, right_f

        self.fn=0

        with gil:

            n_regions = self.preg.shape[1]
            left_region = np.ones(self.n_samples, dtype=np.float64)
            right_region = np.ones(self.n_samples, dtype=np.float64)
            #y = <DOUBLE_t[self.n_samples]> self.y # @debug multiple output
            y = self._get_y_ndarray()[:self.n_samples]

            for i in range(self.n_samples):
                pb_l = 1
                pb_r = 1
                for f in range(self.n_features):

                    sigma = self.sigmas[f]
                    left_f, right_f = self.region_bounds[f]
                    if f == feature:
                        _pb_l = prob_region(self.X[self.X_sample_stride*i + f*self.X_feature_stride],
                                          left_f, threshold, sigma)
                        _pb_r = prob_region(self.X[self.X_sample_stride*i + f*self.X_feature_stride], 
                                          threshold, right_f, sigma)

                        pb_l = pb_l * _pb_l
                        pb_r = pb_r * _pb_r
                    else:
                        _pb_l = prob_region(self.X[self.X_sample_stride*i + f*self.X_feature_stride], 
                                          left_f, right_f, sigma)
                    
                        pb_l = pb_l * _pb_l
                        pb_r = pb_r * _pb_l

                # compute P_left
                left_region[i] = pb_l
                # compute P_right
                right_region[i] = pb_r


            # update preg
            self.preg[:, self.region] = left_region
            self.preg = np.insert(self.preg, n_regions, right_region, axis=1)

            # memoryview don't support dot product
            preg = np.asarray(self.preg)
            gmma = np.linalg.pinv(preg.T.dot(preg)).dot(preg.T).dot(y)
            self.gmma = gmma

            for i in range(self.n_samples):
                self.fn += ( self.y[i] - preg[i].dot(gmma) )**2
        
            self.fn = self.fn / self.n_samples


    cdef int update3(self, SIZE_t new_pos, SIZE_t feature, DOUBLE_t threshold) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef SIZE_t n_regions
        cdef double[:] left_region
        cdef double[:] right_region
        cdef double[:] y

        # cdef time_t t

        cdef int f, i
        cdef double _pb_r, _pb_l, pb_r, pb_l, sigma, left_f, right_f

        self.fn=0
        cdef SIZE_t important_features[100]
        for i in range(100):
            important_features[i] = -1
        important_features[0] = feature
        i = 1

        with gil:
            for f in range(self.n_features):
                sigma = self.sigmas[f]
                if f == feature or sigma == 0:
                    continue
                left_f, right_f = self.region_bounds[f]

                ###########################################################################
                # Huge modification
                ###########################################################################
                # New
                if left_f == -INFINITY and right_f == INFINITY:
                    continue
                important_features[i] = f
                i = i + 1
                # printf('%f %f %d\n', left_f, right_f, f)
                # Old
                # if _pb_l < 1:
                #     important_features[i] = f
                #     i = i + 1
                ###########################################################################

            n_regions = self.preg.shape[1]
            left_region = np.ones(self.n_samples, dtype=np.float64)
            right_region = np.ones(self.n_samples, dtype=np.float64)
            y = self._get_y_ndarray()[:self.n_samples]
            # printf('samples n_top %d %d %f\n', self.n_samples, n_top_feat, threshold)
            for i in range(self.n_samples):
                pb_l = 1
                pb_r = 1
                for f in range(100):
                    if important_features[f] == -1:
                        break
                    sigma = self.sigmas[important_features[f]]
                    left_f, right_f = self.region_bounds[important_features[f]]
                    if important_features[f] == feature:
                        _pb_l = prob_region(self.X[self.X_sample_stride*i + important_features[f]*self.X_feature_stride],
                                          left_f, threshold, sigma)
                        _pb_r = prob_region(self.X[self.X_sample_stride*i + important_features[f]*self.X_feature_stride],
                                          threshold, right_f, sigma)
                        # printf('%f %f %f\n', _pb_l, _pb_r, _pb_r + _pb_l)
                        pb_l = pb_l * _pb_l
                        pb_r = pb_r * _pb_r
                    else:
                        _pb_l = prob_region(self.X[self.X_sample_stride*i + important_features[f]*self.X_feature_stride],
                                          left_f, right_f, sigma)

                        pb_l = pb_l * _pb_l
                        pb_r = pb_r * _pb_l
                # compute P_left
                left_region[i] = pb_l
                # compute P_right
                right_region[i] = pb_r

            # update preg
            self.preg[:, self.region] = left_region
            self.preg = np.insert(self.preg, n_regions, right_region, axis=1)

            # memoryview don't support dot product
            preg = np.asarray(self.preg)

            if self._alpha < 0:
                gmma = np.linalg.pinv(preg.T.dot(preg)).dot(preg.T).dot(y)
            else:

                # Quantile Test
                i_plus = np.identity(self.n_samples)
                i_minus = -1 * np.identity(self.n_samples)
                P_minus = np.multiply(self.preg, -1)
                A = np.concatenate([self.preg.T, P_minus.T, i_plus, i_minus])
                A = A.T
                C_length = self.preg.shape[1] * 2
                C = [0 for _ in range(C_length)]
                C.extend([self._alpha for _ in range(self.n_samples)])
                C.extend([1 - self._alpha for _ in range(self.n_samples)])
                C = np.array(C)
                c_i_bounds = (0, None)
                simplex = linprog(c=C.T, A_eq=A, b_eq=y, bounds=c_i_bounds, method='interior-point', options={'lstsq': True})
                gmma = simplex['x'][:self.preg.shape[1]] - simplex['x'][self.preg.shape[1] + self.preg.shape[1]]
                # End of Quantile Test

            self.gmma = gmma

            for i in range(self.n_samples):
                self.fn += ( self.y[i] - preg[i].dot(gmma) )**2
            self.fn = self.fn / self.n_samples

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        return 1e10


    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:

        impurity_left[0] = self.fn
        impurity_right[0] = self.fn



    #cdef void node_value(self, double* dest) nogil:
    #    """Compute the node value of samples[start:end] into dest."""

    #    pass

    cdef double impurity_improvement(self, double impurity) nogil:

        #if impurity > self.fn:
        #    return impurity - self.fn
        #else:
        #    return self.fn - impurity
        #return self.fn - impurity
        return impurity - self.fn


    cdef double proxy_impurity_improvement2(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)
