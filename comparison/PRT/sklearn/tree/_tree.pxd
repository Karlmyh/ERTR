# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t parent                        # if of the parent node (-1) if root
    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

    bint is_left
    SIZE_t depth # It's not used..could be dropped
    Coord* path

cdef struct Coord:
    bint is_left
    SIZE_t feature
    DOUBLE_t threshold
    bint is_root
    bint is_end


cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef DOUBLE_t* sigmas                # tolerance for X[:,k]
    cdef SIZE_t* samples                   # tolerance for X[:,k]
    cdef DOUBLE_t _alpha                # Loss quantile parameter
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    cdef SIZE_t n_samples   # X.shape[0]
    cdef SIZE_t n_regions   # Number of leaf nodes
    cdef DOUBLE_t* preg     # array (n_samples, n_regions)
    cdef DOUBLE_t* gmma     # array (n_regions)

    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_feature_stride
    cdef DOUBLE_t* y # ignore stride


    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_samples) nogil except -1
    cdef SIZE_t _update_node(self, SIZE_t node_id, SIZE_t parent, bint is_left, bint is_leaf,
                             SIZE_t feature, double threshold, double impurity,
                             SIZE_t n_node_samples,
                             double weighted_n_samples) nogil except -1
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)
    cdef np.ndarray _get_preg_ndarray(self)
    cdef np.ndarray _get_gmma_ndarray(self)

    cdef np.ndarray _get_y_ndarray(self)

    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray predict2(self, object X)
    cpdef np.ndarray predict3(self, object X, object F)
    cpdef np.ndarray quantile_predict(self, object X, object F)
    cpdef np.ndarray predict4(self, object X, object F)
    cpdef np.ndarray predict5(self, object X, object F)
    cpdef set_preg(self, np.ndarray[DOUBLE_t, ndim=2] prego)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply_dense(self, object X)
    cdef np.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)

    cpdef compute_feature_importances(self, normalize=*)

    cdef int _get_parent_path(self, Coord* path, SIZE_t parent, SIZE_t depth, bint is_left)  nogil except -1
    cdef int extra_init(self, SIZE_t X_sample_stride, SIZE_t X_feature_stride, DOUBLE_t* y)
    cdef int _compute_preg(self, DOUBLE_t* preg, DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose)
    cdef int _compute_preg2(self, DOUBLE_t* preg, DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features)
    cdef int _compute_preg3(self, DOUBLE_t* preg, DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features)
    cdef int _compute_preg4(self, DOUBLE_t* preg, DTYPE_t* X, SIZE_t X_sample_stride, SIZE_t X_feature_stride, SIZE_t n_samples, int verbose, SIZE_t* top_features)

    cdef SIZE_t extra_add(self, SIZE_t parent, bint is_left, bint is_leaf, SIZE_t feature, double threshold, double impurity, SIZE_t n_node_samples, SIZE_t left_child, SIZE_t right_child, SIZE_t node_id) nogil except -1
    cdef int extra_copy(self, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples, np.ndarray nodes, SIZE_t max_depth, np.ndarray sigmas)
    cdef int prune(self, SIZE_t n_samples, SIZE_t n_regions, np.ndarray[DOUBLE_t, ndim=2] preg, np.ndarray samples, np.ndarray nodes, SIZE_t max_depth, SIZE_t first_node, SIZE_t second_node, np.ndarray y)
    cpdef int update_gmma(self, np.ndarray new_gmma)

    cpdef yaaa(self)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter              # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_split
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray X_idx_sorted=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)


