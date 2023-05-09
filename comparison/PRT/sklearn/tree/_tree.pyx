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

from cpython cimport Py_INCREF, PyObject

from libc.stdlib cimport free, malloc
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdio cimport printf #

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.optimize import linprog

from ._utils cimport Stack
from ._utils cimport Queue
from ._utils cimport StackRecord
from ._utils cimport PriorityHeap
from ._utils cimport PriorityHeapRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

#
# _arr_lib import
#
from ._arr_lib cimport cydot



cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})



#from libcpp.vector cimport vector

# copy declarations from libcpp.vector to allow nogil
#cdef extern from "<vector>" namespace "std":
#    cdef cppclass vector[T]:
#        void push_back(T&) nogil
#        size_t size()
#        T& operator[](size_t)

import scipy.stats as stats
import numpy as np


cdef inline feat_bound(Coord* path, SIZE_t feature):
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


def compute_gamma(np.ndarray preg, np.ndarray y):
    gmma = np.linalg.pinv(preg.T.dot(preg)).dot(preg.T).dot(y)
    return gmma



# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        tree.extra_init(splitter.X_sample_stride, splitter.X_feature_stride, splitter.y)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        cdef Coord* curr_path = NULL


        # @Debug
        cdef SIZE_t n_samples = splitter.n_samples
        cdef SIZE_t n_outputs = tree.n_outputs
        P_reg_np = np.zeros((n_samples, n_outputs), dtype=np.float64)
        P_reg_T_np = P_reg_np.T
        cdef double[:,:] P_reg = P_reg_np
        cdef double[:,:] P_reg_T = P_reg_np.T
        # print('shape %d, %d' %  (P_reg_np.shape[0], P_reg_np.shape[1]))
        print('shape T %d, %d' %  (P_reg_T_np.shape[0], P_reg_T_np.shape[1]))
        cdef np.ndarray[double, ndim=1] R_temp

        cdef int s_current_node = 0
        cdef SIZE_t features_limit = 100
        cdef SIZE_t top_features[100]
        cdef double alpha = 0.95
        cdef double beta = 2
        cdef double depth_prior
        cdef SIZE_t node_prior

        for index in range(features_limit):
            top_features[index] = -1

        with nogil:

            # @Debug
            # printf('shape %d %d\n', P_reg.shape[0], P_reg.shape[1])
            # printf('shape T %d %d \n', P_reg_T.shape[0], P_reg_T.shape[1])
            #with gil: # need to acuire the gil for "python operation"
            #    R_temp = np.empty(n_samples, dtype=np.double)
            #    print(P_reg[0], P_reg[1], R_temp[1])
            #    R_temp[1] = cydot(P_reg[0], P_reg[1],1)
            #    print(R_temp[1])
            #

            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                curr_path = <Coord *> malloc((depth or 1) * sizeof(Coord))
                tree._get_parent_path(curr_path, parent, depth, is_left)

                splitter.node_reset(start, end, &weighted_n_node_samples, curr_path, 0)

                #################################################################################
                with gil:
                    depth_prior = alpha / ((1 + depth) ** beta)
                    node_prior = np.random.binomial(n=1, p=depth_prior)
                    printf('depth %d, depth_prior %f, b_prior %f', depth, depth_prior, node_prior)
                #################################################################################

                is_leaf = (depth >= max_depth or node_prior == 0 or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))


                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                    #############################################################
                    if not is_leaf:

                        top_features[s_current_node] = split.feature
                        s_current_node = s_current_node + 1
                        if s_current_node == features_limit:
                            s_current_node = s_current_node - 1
                    #############################################################

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)



                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break


                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)


                if not is_leaf:

                    #free(curr_path)

                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break
                else:
                    # save parent path for current leaf
                    tree.nodes[node_id].path = curr_path
                    tree.nodes[node_id].depth = depth


                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

            # end of while

        # end of nogil

        ############################################################################################
        for s_current_node in range(tree.n_regions - 1):
            for k in range(s_current_node+1, tree.n_regions - 1):
                if top_features[s_current_node] == top_features[k]:
                    top_features[k] = -1
        ############################################################################################

        # For prediction create Pr matrix for X
        safe_realloc(&tree.preg, n_samples * tree.n_regions)
        tree._compute_preg2(tree.preg, splitter.X, splitter.X_sample_stride, splitter.X_feature_stride, tree.n_samples, 1, top_features)

        # Compute sigma
        cdef np.ndarray gmma
        gmma = compute_gamma(tree._get_preg_ndarray()[:tree.n_samples],
                             tree._get_y_ndarray()[:tree.n_samples])

        safe_realloc(&tree.gmma, tree.n_regions)
        #tree.gmma = <DOUBLE_t*>gmma.data
        memcpy(tree.gmma, gmma.data, sizeof(DOUBLE_t)*tree.n_regions)


        if rc == -1:
            raise MemoryError()


cdef class DepthFirstTreeBuilderOrig(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples, NULL, 0)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


# breadth first builder ---------------------------------------------------------

cdef class BreadthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)
        splitter.extra_init(X, tree.sigmas, tree._alpha)
        tree.extra_init(splitter.X_sample_stride, splitter.X_feature_stride, splitter.y)
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        #cdef Stack queue = Stack(INITIAL_STACK_SIZE)
        cdef Queue queue = Queue(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        cdef Coord* curr_path = NULL

        # @Debug
        cdef SIZE_t n_samples = splitter.n_samples
        cdef SIZE_t n_outputs = tree.n_outputs
        # P_reg_np = np.zeros((n_samples, n_outputs), dtype=np.float64)
        # P_reg_T_np = P_reg_np.T
        # cdef double[:,:] P_reg = P_reg_np
        # cdef double[:,:] P_reg_T = P_reg_np.T
        # print('shape %d, %d' %  (P_reg_np.shape[0], P_reg_np.shape[1]))
        cdef np.ndarray[double, ndim=1] R_temp

        cdef int s_current_node = 0
        cdef SIZE_t features_limit = 100
        cdef SIZE_t top_features[100]
        cdef double alpha = 0.95
        cdef double beta = 2
        cdef double depth_prior
        cdef SIZE_t node_prior

        for index in range(features_limit):
            top_features[index] = -1
        np.random.seed(42)

        with nogil:

            # @Debug
            #with gil: # need to acuire the gil for "python operation"
            #    R_temp = np.empty(n_samples, dtype=np.double)
            #    print(P_reg[0], P_reg[1], R_temp[1])
            #    R_temp[1] = cydot(P_reg[0], P_reg[1],1)
            #    print(R_temp[1])

            # push root node onto stack
            rc = queue.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()
            while not queue.is_empty():
                queue.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                curr_region = stack_record.curr_region

                n_node_samples = end - start
                curr_path = <Coord *> malloc((depth+1) * sizeof(Coord))
                tree._get_parent_path(curr_path, parent, depth, is_left)

                splitter.node_reset(start, end, &weighted_n_node_samples, curr_path, curr_region)
                ######################################################################################
                with gil:
                    depth_prior = alpha / ((1 + depth) ** beta)
                    node_prior = np.random.binomial(n=1, p=depth_prior)
                ######################################################################################

                is_leaf = (depth >= max_depth or # node_prior == 0 or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf)

                ##or weighted_n_node_samples < 2 * min_weight_leaf) # Ignore weight here


                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)

                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18

                    #with gil:
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                    #############################################################
                    if not is_leaf:

                        top_features[s_current_node] = split.feature
                        s_current_node = s_current_node + 1
                        if s_current_node == features_limit:
                            s_current_node = s_current_node - 1
                    #############################################################

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)


                if not is_leaf:

                    #free(curr_path)

                    # Push left child on stack
                    rc = queue.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features, curr_region)
                    if rc == -1:
                        break

                    # Push right child on stack
                    rc = queue.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features, tree.n_regions-1)

                    if rc == -1:
                        break
                else:
                    # save parent path for current leaf
                    tree.nodes[node_id].path = curr_path
                    tree.nodes[node_id].depth = depth


                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

            # end of while

        # end of nogil

        ############################################################################################
        for s_current_node in range(tree.n_regions - 1):
            for k in range(s_current_node+1, tree.n_regions - 1):
                if top_features[s_current_node] == top_features[k]:
                    top_features[k] = -1
        ############################################################################################

        # For prediction create Pr matrix for X
        safe_realloc(&tree.preg, n_samples * tree.n_regions)
        tree._compute_preg2(tree.preg, splitter.X, splitter.X_sample_stride, splitter.X_feature_stride, tree.n_samples, 1, top_features)

        cdef np.ndarray gmma
        # Compute normal gmma

        if tree._alpha < 0:
            gmma = compute_gamma(tree._get_preg_ndarray()[:tree.n_samples],
                                 tree._get_y_ndarray()[:tree.n_samples])
        else:
            # Quantile Gamma
            temp_preg = tree._get_preg_ndarray()
            i_plus = np.identity(tree.n_samples)
            i_minus = -1 * np.identity(tree.n_samples)
            P_minus = np.multiply(temp_preg, -1)
            A = np.concatenate([temp_preg.T, P_minus.T, i_plus, i_minus])
            A = A.T
            C_length = temp_preg.shape[1] * 2
            C = [0 for _ in range(C_length)]
            C.extend([tree._alpha for _ in range(tree.n_samples)])
            C.extend([1 - tree._alpha for _ in range(tree.n_samples)])
            C = np.array(C)
            c_i_bounds = (0, None)
            simplex = linprog(c=C.T, A_eq=A, b_eq=y, bounds=c_i_bounds, method='interior-point', options={'lstsq': True})
            gmma = simplex['x'][:temp_preg.shape[1]] - simplex['x'][temp_preg.shape[1]: C_length]
            # End of Quantile Test

        safe_realloc(&tree.gmma, tree.n_regions)
        memcpy(tree.gmma, gmma.data, sizeof(DOUBLE_t)*tree.n_regions)

        for index in range(n_samples):
            tree.samples[index] = splitter.samples[index]

        if rc == -1:
            raise MemoryError()

    cpdef extra_copy(self, Tree tree, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples, np.ndarray nodes, SIZE_t max_depth, np.ndarray sigmas):

        self.splitter.criterion.set_preg(preg)
        tree.extra_copy(n_samples, n_regions, preg, samples, nodes, max_depth, sigmas)

    cpdef prune(self, Tree tree, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples, np.ndarray nodes, SIZE_t max_depth, SIZE_t first_node, SIZE_t second_node, np.ndarray y):

        self.splitter.criterion.set_preg(preg)
        tree.prune(n_samples, n_regions, preg, samples, nodes, max_depth, first_node, second_node, y)

    cpdef int update_gmma(self, Tree tree, np.ndarray new_gmma):

        tree.update_gmma(new_gmma)

    cpdef expand(self, Tree tree, object X, np.ndarray y, SIZE_t start, SIZE_t end, SIZE_t parent,
                  double impurity, SIZE_t depth, SIZE_t curr_region, SIZE_t nt_features, np.ndarray t_features,
                  SIZE_t temp_node_id, bint is_left, SIZE_t inp_samples, bint expansion, SIZE_t rand_p, double rand_s,
                  np.ndarray sample_weight=None,
                  np.ndarray X_idx_sorted=None, np.ndarray preg=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        # WATCH IT HEREEEEEEEEEEEEEEEEEEEEEEE
        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter

        cdef bint growable_node = 0
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        splitter.extra_init(X, tree.sigmas, tree._alpha)

        tree.extra_init(splitter.X_sample_stride, splitter.X_feature_stride, splitter.y)

        cdef SIZE_t n_node_samples = inp_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id = temp_node_id

        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = tree.max_depth
        cdef int rc = 0

        cdef Queue queue = Queue(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        cdef Coord* curr_path = NULL

        # @Debug
        cdef SIZE_t n_samples = inp_samples
        cdef SIZE_t n_outputs = tree.n_outputs

        cdef int expand_ind = 0
        cdef SIZE_t features_limit = 100
        cdef SIZE_t top_features[100]
        cdef Node* node

        for index in range(features_limit):
            top_features[index] = -1
        for index in range(nt_features):
            top_features[index] = t_features[index]

        if impurity == -1:
            impurity = splitter.node_impurity()

        np.random.seed(42)

        if parent == -1:
            parent = _TREE_UNDEFINED

        # Don't know why it doesn't work FOR NOW
        # splitter.samples = tree.samples
        for index in range(X.shape[0]):
            splitter.samples[index] = tree.samples[index]

        if expansion:
            splitter.criterion.set_preg(preg)

        with nogil:

            # push root node onto stack
            rc = queue.push(start, end, depth, parent, is_left, impurity, 0, curr_region)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()
            while not queue.is_empty():
                queue.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                curr_region = stack_record.curr_region

                n_node_samples = end - start
                curr_path = <Coord *> malloc((depth+1) * sizeof(Coord))
                tree._get_parent_path(curr_path, parent, depth, is_left)

                # printf('start %d end %d depth %d parent %d isleft %d impurity %f region %d\n', start, end, depth, parent, is_left, impurity, curr_region)

                splitter.node_reset(start, end, &weighted_n_node_samples, curr_path, curr_region)
                is_leaf = (depth >= max_depth or expand_ind > 0 or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf)
                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))
                if not is_leaf:
                    split.feature = rand_p
                    split.threshold = rand_s
                    splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                if (expand_ind > 0) or (expansion == 0):
                    node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                             split.threshold, impurity, n_node_samples,
                                             weighted_n_node_samples)
                else:
                    tree._update_node(node_id, parent, is_left, is_leaf, split.feature, split.threshold, impurity,
                                      n_node_samples, weighted_n_node_samples)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:

                    #free(curr_path)

                    growable_node = 1
                    # Push left child on stack
                    rc = queue.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features, curr_region)
                    if rc == -1:
                        break

                    # Push right child on stack
                    rc = queue.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features, tree.n_regions-1)

                    if rc == -1:
                        break
                else:
                    # save parent path for current leaf
                    tree.nodes[node_id].path = curr_path
                    tree.nodes[node_id].depth = depth

                if depth > max_depth_seen:
                    max_depth_seen = depth
                expand_ind += 1

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

            # end of while

        # end of nogil

        cdef np.ndarray gmma
        if growable_node:
            top_features[tree.n_regions - 2] = split.feature
            ############################################################################################
            for index in range(tree.n_regions - 1):
                for k in range(index+1, tree.n_regions - 1):
                    if top_features[index] == top_features[k]:
                        top_features[k] = -1
            ############################################################################################

            safe_realloc(&tree.preg, n_samples * tree.n_regions)
            tree._compute_preg3(tree.preg, splitter.X, splitter.X_sample_stride, splitter.X_feature_stride, tree.n_samples, 1, top_features)

            gmma = compute_gamma(tree._get_preg_ndarray()[:tree.n_samples], tree._get_y_ndarray()[:tree.n_samples])

            safe_realloc(&tree.gmma, tree.n_regions)
            memcpy(tree.gmma, gmma.data, sizeof(DOUBLE_t) * tree.n_regions)

            for index in range(n_samples):
                tree.samples[index] = splitter.samples[index]

            tree.set_preg(self.splitter.criterion.get_preg())

        cdef np.ndarray[DOUBLE_t, ndim=2] preg_ones
        if tree.node_count ==1:
            preg_ones = np.ones((n_samples, 1), dtype=np.float64)
            safe_realloc(&tree.preg, n_samples * tree.n_regions)
            tree.set_preg(preg_ones)
            gmma = compute_gamma(preg_ones, tree._get_y_ndarray()[:tree.n_samples])

            safe_realloc(&tree.gmma, tree.n_regions)
            memcpy(tree.gmma, gmma.data, sizeof(DOUBLE_t) * tree.n_regions)


        if rc == -1:
            raise MemoryError()



# Best first builder ----------------------------------------------------------

cdef inline int _add_to_frontier(PriorityHeapRecord* rec,
                                 PriorityHeap frontier) nogil except -1:
    """Adds record ``rec`` to the priority queue ``frontier``

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    return frontier.push(rec.node_id, rec.start, rec.end, rec.pos, rec.depth,
                         rec.is_leaf, rec.improvement, rec.impurity,
                         rec.impurity_left, rec.impurity_right)


cdef class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf,  min_weight_leaf,
                  SIZE_t max_depth, SIZE_t max_leaf_nodes,
                  double min_impurity_decrease, double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef PriorityHeap frontier = PriorityHeap(INITIAL_STACK_SIZE)
        cdef PriorityHeapRecord record
        cdef PriorityHeapRecord split_node_left
        cdef PriorityHeapRecord split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        # Initial capacity
        cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(splitter, tree, 0, n_node_samples,
                                      INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                                      &split_node_left)
            if rc >= 0:
                rc = _add_to_frontier(&split_node_left, frontier)

            if rc == -1:
                with gil:
                    raise MemoryError()

            while not frontier.is_empty():
                frontier.pop(&record)

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    rc = self._add_split_node(splitter, tree,
                                              record.start, record.pos,
                                              record.impurity_left,
                                              IS_NOT_FIRST, IS_LEFT, node,
                                              record.depth + 1,
                                              &split_node_left)
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    rc = self._add_split_node(splitter, tree, record.pos,
                                              record.end,
                                              record.impurity_right,
                                              IS_NOT_FIRST, IS_NOT_LEFT, node,
                                              record.depth + 1,
                                              &split_node_right)
                    if rc == -1:
                        break

                    # Add nodes to queue
                    rc = _add_to_frontier(&split_node_left, frontier)
                    if rc == -1:
                        break

                    rc = _add_to_frontier(&split_node_right, frontier)
                    if rc == -1:
                        break

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(self, Splitter splitter, Tree tree,
                                    SIZE_t start, SIZE_t end, double impurity,
                                    bint is_first, bint is_left, Node* parent,
                                    SIZE_t depth,
                                    PriorityHeapRecord* res) nogil except -1:
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        cdef SplitRecord split
        cdef SIZE_t node_id
        cdef SIZE_t n_node_samples
        cdef SIZE_t n_constant_features = 0
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split
        cdef double weighted_n_node_samples
        cdef bint is_leaf
        cdef SIZE_t n_left, n_right
        cdef double imp_diff

        splitter.node_reset(start, end, &weighted_n_node_samples, NULL, 0)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth > self.max_depth or
                   n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf or
                   impurity <= min_impurity_split)

        if not is_leaf:
            splitter.node_split(impurity, &split, &n_constant_features)
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split.feature, split.threshold, impurity, n_node_samples,
                                 weighted_n_node_samples)
        if node_id == <SIZE_t>(-1):
            return -1

        # compute values also for split nodes (might become leafs later).
        splitter.node_value(tree.value + node_id * tree.value_stride)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        return 0


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """

    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property samples:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.samples, self.n_samples)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property preg:
        def __get__(self):
            return self._get_preg_ndarray()[:self.n_samples]
    property gmma:
        def __get__(self):
            return self._get_gmma_ndarray()[:self.n_regions]

    property y:
        def __get__(self):
            return self._get_y_ndarray()[:self.n_samples]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_samples, int n_outputs, np.ndarray[DOUBLE_t, ndim=1] sigmas, double _alpha,
                  np.ndarray[SIZE_t, ndim=1] samples):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.samples = NULL
        safe_realloc(&self.samples, n_samples)

        self.sigmas = NULL
        safe_realloc(&self.sigmas, n_features)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        self.n_regions = 1

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        for k in range(n_samples):
            self.samples[k] = samples[k]

        for k in range(n_features):
            self.sigmas[k] = sigmas[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

        self.X = NULL # free in splitter
        #self.samples = NULL # free in splitter
        self.X_sample_stride = 0
        self.X_feature_stride = 0


        self.y = NULL
        self.preg = NULL
        self.gmma = NULL
        self._alpha = _alpha

    cdef int extra_init(self, SIZE_t X_sample_stride, SIZE_t X_feature_stride,
                       DOUBLE_t* y):
        ''' Extra Info to access data. '''
        self.X_sample_stride = X_sample_stride
        self.X_feature_stride = X_feature_stride

        # I don't get why the first value of y (y[O] is lost
        # when y is not deep copied ?!
        safe_realloc(&self.y, self.n_samples)
        memcpy(self.y, y, sizeof(DOUBLE_t)*self.n_samples)
        #self.y = y

        return 0

    cdef int extra_copy(self, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples, np.ndarray nodes, SIZE_t max_depth, np.ndarray sigmas):

        cdef int i=0
        cdef int j=0
        cdef int k=0

        for i in range(self.n_features):
            self.sigmas[i] = sigmas[i]
        self.n_regions = n_regions
        self.max_depth = max_depth
        for k in range(n_samples):
            self.samples[k] = samples[k]

        safe_realloc(&self.preg, n_samples * n_regions)
        for i in range(n_samples):
            for j in range(n_regions):
                self.preg[i*n_regions + j] = preg[i, j]

        self.capacity = nodes.shape[0]
        self._resize_c(self.capacity)

        memcpy(self.nodes, (<np.ndarray> nodes).data, self.capacity * sizeof(Node))
        self.node_count = self.capacity
        if self.node_count > 1:
            for nid in range(self.node_count):
                node = &self.nodes[nid]
                if node.left_child == _TREE_LEAF:
                    curr_path = <Coord *> malloc((node.depth+1) * sizeof(Coord))
                    self._get_parent_path(curr_path, node.parent, node.depth, node.is_left)
                    node.path = curr_path

    cdef SIZE_t extra_add(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, SIZE_t left_child, SIZE_t right_child, SIZE_t node_id) nogil except -1:

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = n_node_samples
        node.left_child = left_child
        node.right_child = right_child
        node.parent = parent

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        node.is_left = is_left
        node.parent = parent
        node.path = NULL

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        return 0

    cdef int prune(self, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples,
                   np.ndarray nodes, SIZE_t max_depth, SIZE_t first_node, SIZE_t second_node, np.ndarray y):

        cdef int i=0
        cdef int j=0
        cdef int k=0

        self.n_regions = n_regions
        self.max_depth = max_depth
        for k in range(n_samples):
            self.samples[k] = samples[k]

        safe_realloc(&self.preg, n_samples * n_regions)
        for i in range(n_samples):
            for j in range(n_regions):
                self.preg[i*n_regions + j] = preg[i, j]

        self.capacity = nodes.shape[0] - 2
        self._resize_c(self.capacity)
        self.node_count = self.capacity

        cdef Node* temp_nodes
        temp_nodes = <Node *> malloc((nodes.shape[0]) * sizeof(Node))
        memcpy(temp_nodes, (<np.ndarray> nodes).data, nodes.shape[0] * sizeof(Node))

        k = 0
        cdef bint is_leaf
        for nid in range(nodes.shape[0]):
            if nid == first_node or nid == second_node:
                continue
            temp_node = &temp_nodes[nid]
            is_leaf = temp_node.left_child == -1 and temp_node.right_child == -1
            if temp_node.parent > second_node:
                temp_node.parent -= 2
            i = self.extra_add(temp_node.parent, temp_node.is_left, is_leaf, temp_node.feature, temp_node.threshold, temp_node.impurity, temp_node.n_node_samples, temp_node.left_child, temp_node.right_child, k)
            temp_orig_node = &self.nodes[k]
            temp_orig_node.depth = temp_node.depth
            k += 1

        for nid in range(self.node_count):
            node = &self.nodes[nid]
            if node.left_child == _TREE_LEAF and nid != 0:
                curr_path = <Coord *> malloc((node.depth+1) * sizeof(Coord))
                self._get_parent_path(curr_path, node.parent, node.depth, node.is_left)
                node.path = curr_path

        for i in range(nodes.shape[0]):
            free(temp_nodes[i].path)
        free(temp_nodes)
        # free(self.gmma)
        return 0

    cpdef int update_gmma(self, np.ndarray new_gmma):

        safe_realloc(&self.gmma, self.n_regions)
        memcpy(self.gmma, new_gmma.data, sizeof(DOUBLE_t) * self.n_regions)
        return 0


    cpdef set_preg(self, np.ndarray[DOUBLE_t, ndim=2] prego):

        cdef int i=0
        cdef int j=0
        for i in range(self.n_samples):
            for j in range(self.n_regions):
                self.preg[i*self.n_regions + j] = prego[i, j]

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.samples)
        free(self.value)
        free(self.sigmas)

        cdef SIZE_t i
        for i in range(self.node_count):
            free(self.nodes[i].path)
        free(self.nodes)

        free(self.preg)
        free(self.gmma)
        free(self.y)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef int _compute_preg(self, DOUBLE_t* preg,  DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose):
        ''' Warning will not work with multiple output AND extra weight. '''

        #cdef DTYPE_t* X = <DTYPE_t*>X

        cdef SIZE_t n_features = self.n_features

        cdef Node* node = NULL

        # Save this in self to be faster.
        #cdef Coord** paths = <Coord**> malloc(self.n_regions * sizeof(SIZE_t))
        # Memoryview, faster
        cdef  double[:,:,:] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)
        # Old Way
        #cdef  np.ndarray[DOUBLE_t, ndim=3] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)

        cdef DOUBLE_t pb, left_f, right_f, sigma, _pb
        cdef int i,k,f,nid

        # Get path/length of regions
        i = 0
        cdef int cpt_region = 0
        for nid in range(self.node_count):
            node = &self.nodes[nid]
            if node.left_child == _TREE_LEAF:
                ####################################################
                ### Soulager temps calculs en enlevants les prints #NOV 2018
                ####################################################
                ###if verbose == 1:
                    ###printf('Region_%d: ', cpt_region)
                for f in range(n_features):
                    left_f, right_f = feat_bound(node.path, f)
                    regions[i, f, 0] = left_f
                    regions[i, f, 1] = right_f
                    ################################################################
                    ### Soulager temps calculs en enlevants les prints des regions #NOV 2018
                    ################################################################
                    ###if verbose == 1:
                        ###printf('%d: %f -- %f, ', f, left_f, right_f)
                ###if verbose == 1:
                    ###printf('\n')
                cpt_region += 1
                i += 1

        for i in range(n_samples):
            for k in range(self.n_regions):
                pb = 1
                for f in range(n_features):
                    sigma = self.sigmas[f]
                    left_f, right_f = regions[k, f]
                    _pb = prob_region(X[X_sample_stride*i + f*X_feature_stride],
                                      left_f, right_f, sigma)
                    ###############################################################
                    ### Soulager temps de calcul en enlevant les prints des regions #NOV 2018
                    ###############################################################
                    ###printf('%d %d: %d %d:  %f ', i,  X_sample_stride, f, X_feature_stride, X[X_sample_stride*i + f*X_feature_stride])
                    
                    #printf('region %d, feature %d, pr: %f -- left: %f   right: %f\n', k, f, _pb,
                    #      left_f, right_f)

                    #if _pb <= EPSILON:
                    #    _pb = EPSILON
                    pb = pb * _pb
                    ###############################################################
                    ### Soulager temps de calcul en enlevant les prints des regions #NOV 2018
                    ###############################################################
                    ###printf('\n')
                preg[i*self.n_regions + k] = pb
            ### Soulager temps de calcul en enlevant les prints des regions #NOV 2018
            ###printf('\n')
        ###############################################################
        ### Soulager temps de calcul en enlevant les prints des regions #NOV 2018
        ###############################################################
        ###for i in range(n_samples):
            ###for k in range(self.n_regions):
                ###printf('%f ', preg[i*self.n_regions + k])
            ###printf('\n')
        #free(paths)
        return 0


    cdef int _compute_preg2(self, DOUBLE_t* preg,  DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features):
        ''' Warning will not work with multiple output AND extra weight. '''

        #cdef DTYPE_t* X = <DTYPE_t*>X

        cdef SIZE_t n_features = self.n_features

        cdef Node* node = NULL

        # Save this in self to be faster.
        #cdef Coord** paths = <Coord**> malloc(self.n_regions * sizeof(SIZE_t))
        # Memoryview, faster
        cdef  double[:,:,:] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)
        # Old Way
        #cdef  np.ndarray[DOUBLE_t, ndim=3] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)

        cdef DOUBLE_t pb, left_f, right_f, sigma, _pb
        cdef int i,k,f,nid

        # Get path/length of regions
        i = 0
        cdef int cpt_region = 0
        for nid in range(self.node_count):
            node = &self.nodes[nid]
            # printf('%d %d %d %d %f %f %d %f\n', node.parent, node.left_child, node.right_child, node.feature, node.threshold, node.impurity, node.n_node_samples, node.weighted_n_node_samples)
            if node.left_child == _TREE_LEAF:
                ####################################################
                ### Soulager temps calculs en enlevants les prints #NOV 2018
                ####################################################
                ###if verbose == 1:
                    ###printf('Region_%d: ', cpt_region)
                for f in range(n_features):
                    left_f, right_f = feat_bound(node.path, f)
                    regions[i, f, 0] = left_f
                    regions[i, f, 1] = right_f
                    ################################################################
                    ### Soulager temps calculs en enlevants les prints des regions #NOV 2018
                    ################################################################
                    ###if verbose == 1:
                        ###printf('%d: %f -- %f, ', f, left_f, right_f)
                ###if verbose == 1:
                    ###printf('\n')
                cpt_region += 1
                i += 1

        # New Code
        #########################################################################################################################################
        # cdef double quantile = (1 + np.power(0.5, 1/4)) / 2
        # quantile = 2*prob_region(0, -INFINITY, quantile, 1)
        # printf('quantile %f \n', quantile)
        # cdef double* min_regions = [INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY]
        # cdef double nominator
        #########################################################################################################################################

        for i in range(n_samples):
            for k in range(self.n_regions):
                pb = 1
                for f in range(self.n_regions - 1):
                    if top_features[f] == -1:
                        continue
                    sigma = self.sigmas[top_features[f]]
                    left_f, right_f = regions[k, top_features[f]]
                    _pb = prob_region(X[X_sample_stride*i + top_features[f]*X_feature_stride],
                                     left_f, right_f, sigma)
                    pb = pb * _pb
                preg[i*self.n_regions + k] = pb
        return 0

    cdef int _compute_preg3(self, DOUBLE_t* preg,  DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features):
        ''' Warning will not work with multiple output AND extra weight. '''

        cdef SIZE_t n_features = self.n_features

        cdef Node* node = NULL
        # Memoryview, faster
        cdef  double[:,:,:] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)

        cdef DOUBLE_t pb, left_f, right_f, sigma, _pb
        cdef int i,k,f,nid

        # Get path/length of regions
        i = 0
        for nid in range(self.node_count):
            node = &self.nodes[nid]
            if node.left_child == _TREE_LEAF:
                for f in range(n_features):
                    left_f, right_f = feat_bound(node.path, f)
                    regions[i, f, 0] = left_f
                    regions[i, f, 1] = right_f
                i += 1

        for i in range(n_samples):
            for k in range(self.n_regions):
                pb = 1
                for f in range(self.n_regions - 1):
                    if top_features[f] == -1:
                        continue
                    sigma = self.sigmas[top_features[f]]
                    left_f, right_f = regions[k, top_features[f]]
                    _pb = prob_region(X[X_sample_stride*i + top_features[f]*X_feature_stride],
                                     left_f, right_f, sigma)
                    pb = pb * _pb
                preg[i*self.n_regions + k] = pb

        return 0


    cdef int _compute_preg4(self, DOUBLE_t* preg,  DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features):
        ''' Warning will not work with multiple output AND extra weight. '''

        cdef SIZE_t n_features = self.n_features

        cdef Node* node = NULL
        cdef Node* temp_node = NULL
        # Memoryview, faster
        cdef  double[:,:,:] regions = np.zeros((self.n_regions, self.n_features, 2), dtype=np.float64)

        cdef DOUBLE_t pb, left_f, right_f, sigma, _pb
        cdef int i,k,f,nid

        # Get path/length of regions
        i = 0
        for nid in range(self.node_count):
            node = &self.nodes[nid]
            if node.left_child == _TREE_LEAF:
                for f in range(n_features):
                    ###############################################################
                    left_f, right_f = feat_bound(node.path, f)
                    ###############################################################
                    regions[i, f, 0] = left_f
                    regions[i, f, 1] = right_f
                i += 1

        for i in range(n_samples):
            for k in range(self.n_regions):
                pb = 1
                for f in range(self.n_regions - 1):
                    if top_features[f] == -1:
                        continue
                    sigma = self.sigmas[top_features[f]]
                    left_f, right_f = regions[k, top_features[f]]
                    _pb = prob_region(X[X_sample_stride*i + top_features[f]*X_feature_stride],
                                     left_f, right_f, sigma)
                    pb = pb * _pb
                preg[i*self.n_regions + k] = pb

        return 0

    cdef int _get_parent_path(self, Coord* path,  SIZE_t parent_id, SIZE_t depth, bint is_left)  nogil except -1:

        #safe_realloc(&node.path, depth)
        #path = <Coord *> malloc(depth * sizeof(Coord))

        cdef Node* parent
        cdef Coord* coord
        cdef bint curr_is_left = is_left


        if parent_id ==_TREE_UNDEFINED:
            coord  = &path[0]
            coord.is_left = curr_is_left
            coord.is_root = True
            coord.is_end = True
            #coord.feature = _TREE_UNDEFINED
            #coord.threshold = -42
            return 0

        else:
            parent = &self.nodes[parent_id]

        cdef SIZE_t i = 0
        while True:
            coord  = &path[i]
            coord.is_root = False
            coord.is_left = curr_is_left
            coord.feature = parent.feature
            coord.threshold = parent.threshold
            #if i == depth-1:
            if parent.parent == _TREE_UNDEFINED:
                coord.is_end = True
                break
            else:
                coord.is_end = False

            curr_is_left = parent.is_left
            parent = &self.nodes[parent.parent]
            i += 1

        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        node.is_left = is_left
        node.parent = parent
        node.path = NULL

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
            self.n_regions += 1

        self.node_count += 1

        return node_id

    cdef SIZE_t _update_node(self, SIZE_t node_id, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        node.is_left = is_left
        node.parent = parent
        node.path = NULL

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
            self.n_regions += 1

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray predict2(self, object X):
        """Predict target for X based on probabilistic region"""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef int i,k

        # Initialize output
        cdef np.ndarray[DOUBLE_t] predictions_arr = np.zeros((n_samples,), dtype=np.float64)
        cdef DOUBLE_t* predictions = <DOUBLE_t*> predictions_arr.data

        cdef DOUBLE_t* preg_x = NULL
        safe_realloc(&preg_x, n_samples * self.n_regions)

        self._compute_preg(preg_x, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0)

        #self.X_feature_stride = n_samples
        
        for i in range(n_samples):

            for k in range(self.n_regions):

                #printf('final: x%d:%f,  gmma%d:%f, pr:%f \n', i,
                #       X_ptr[X_sample_stride*i],
                #       k,  self.gmma[k],  preg_x[i*self.X_sample_stride + k])
                
                #Test prediction values
                ##printf('(%d, %d, %f)\n', i, k, self.gmma[k]) 
                predictions[i] += self.gmma[k] * preg_x[i*self.n_regions + k]
                ##printf('temp prediction %f\n', predictions[i])
                #Test prediction values
            ##printf('predictions_i for i %d prediction %f\n', i,   predictions[i])
        return predictions_arr

    cpdef np.ndarray predict3(self, object X, object F):
        """Predict target for X based on probabilistic region"""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef double proba_curr
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data

        cdef np.ndarray F_ndarray = F
        cdef SIZE_t* F_ptr = <SIZE_t*> F_ndarray.data

        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef int i,k

        # Initialize output
        cdef np.ndarray[DOUBLE_t] predictions_arr = np.zeros((n_samples,), dtype=np.float64)
        cdef DOUBLE_t* predictions = <DOUBLE_t*> predictions_arr.data

        cdef DOUBLE_t* preg_x = NULL
        safe_realloc(&preg_x, n_samples * self.n_regions)
        self._compute_preg2(preg_x, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0, F_ptr)

        for i in range(n_samples):

            for k in range(self.n_regions):

                # proba_curr = self.gmma[k] * preg_x[i*self.n_regions + k]
                proba_curr = self.gmma[k] * preg_x[i*self.n_regions + k]
                predictions[i] += proba_curr
        return predictions_arr

    cpdef np.ndarray quantile_predict(self, object X, object F):
        """Predict target for X based on probabilistic region"""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef double proba_curr
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data

        cdef np.ndarray F_ndarray = F
        cdef SIZE_t* F_ptr = <SIZE_t*> F_ndarray.data

        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef int i,k

        # Initialize output
        cdef np.ndarray[DOUBLE_t] predictions_arr = np.zeros((n_samples,), dtype=np.float64)
        cdef DOUBLE_t* predictions = <DOUBLE_t*> predictions_arr.data

        safe_realloc(&self.preg, n_samples * self.n_regions)
        self._compute_preg2(self.preg, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0, F_ptr)

        for i in range(n_samples):

            for k in range(self.n_regions):

                # proba_curr = self.gmma[k] * preg_x[i*self.n_regions + k]
                proba_curr = self.gmma[k] * self.preg[i*self.n_regions + k]
                predictions[i] += proba_curr
        return predictions_arr

    cpdef np.ndarray predict4(self, object X, object F):
        """Predict target for X based on probabilistic region"""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data

        cdef np.ndarray F_ndarray = F
        cdef SIZE_t* F_ptr = <SIZE_t*> F_ndarray.data

        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef int i,k

        # Initialize output
        cdef np.ndarray[DOUBLE_t] predictions_arr = np.zeros((n_samples,), dtype=np.float64)
        cdef DOUBLE_t* predictions = <DOUBLE_t*> predictions_arr.data

        cdef DOUBLE_t* preg_x = NULL
        safe_realloc(&preg_x, n_samples * self.n_regions)
        self._compute_preg4(preg_x, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0, F_ptr)

        #self.X_feature_stride = n_samples
        for i in range(n_samples):
            for k in range(self.n_regions):

                proba_curr = self.gmma[k] * preg_x[i*self.n_regions + k]
                predictions[i] += proba_curr

        return predictions_arr

    cpdef np.ndarray predict5(self, object X, object F):
        """Predict target for X based on probabilistic region"""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data

        cdef np.ndarray F_ndarray = F
        cdef SIZE_t* F_ptr = <SIZE_t*> F_ndarray.data

        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        cdef int i,k

        # Initialize output
        cdef np.ndarray[DOUBLE_t] predictions_arr = np.zeros((n_samples,), dtype=np.float64)
        cdef DOUBLE_t* predictions = <DOUBLE_t*> predictions_arr.data

        # cdef DOUBLE_t* preg_x = NULL
        # safe_realloc(&preg_x, n_samples * self.n_regions)
        # self._compute_preg4(preg_x, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0, F_ptr)

        safe_realloc(&self.preg, n_samples * self.n_regions)
        self._compute_preg4(self.preg, X_ptr, X_sample_stride, X_fx_stride, n_samples, 0, F_ptr)

        for i in range(n_samples):
            for k in range(self.n_regions):

                proba_curr = self.gmma[k] * self.preg[i*self.n_regions + k]
                # proba_curr = self.gmma[k] * preg_x[i*self.n_regions + k]
                predictions[i] += proba_curr

        return predictions_arr

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef Coord* path

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # @Debug => check the path of leaf nodes.
                #path = node.path
                #for j in range(node.depth):
                #    printf('path:%d -  %d, %d, %f\n', j, path[j].feature,
                #          path[j].is_left, path[j].threshold)

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out


    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out


    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cpdef yaaa(self):


        cdef int o
        printf('regions: %d\n', self.n_regions)
        for o in range(self.n_regions):
            printf('taaa %f\n', self.gmma[o])


    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_preg_ndarray(self):
        """Wraps value as a 2-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """

        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.n_samples
        shape[1] = <np.npy_intp> self.n_regions
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, self.preg)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_gmma_ndarray(self):
        """Wraps value as a 1-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.n_regions
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(DOUBLE_t)
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.gmma)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

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
