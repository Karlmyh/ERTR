"""Gradient Boosted Regression Trees

This module contains methods for fitting gradient boosted regression trees for
both classification and regression.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  regression problems.
"""

# Authors: Peter Prettenhofer, Scott White, Gilles Louppe, Emanuele Olivetti,
#          Arnaud Joly, Jacob Schreiber
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from .base import BaseEnsemble
from ..base import RegressorMixin
from ..externals import six

from ._gradient_boosting import predict_stages
from ._gradient_boosting import predict_stage

import numbers
import random
import numpy as np
import copy
import math
from scipy.stats import invgamma
from scipy.stats import norm
from scipy.stats import chi2
# import statsmodels.api as sm

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy import stats

from time import time
from ..model_selection import train_test_split
from ..tree.tree import DecisionTreeRegressor
from ..tree.tree import TreeHelper
from ..tree._tree import DTYPE
from ..tree._tree import TREE_LEAF
from ..linear_model import LinearRegression as lr

from ..utils import check_random_state
from ..utils import check_array
from ..utils import check_X_y
from ..utils import check_consistent_length
from ..utils import deprecated
from ..utils.validation import check_is_fitted

__all__ = ["BARTRegressor"]

MAX_INT = np.iinfo(np.int32).max


class ZeroEstimator(object):
    """An estimator that simply predicts zero. """

    def fit(self, X, y, sample_weight=None):
        self.n_classes = 1

    def predict(self, X):
        check_is_fitted(self, 'n_classes')

        y = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        y.fill(0.0)
        return y


class MeanEstimator(object):
    """An estimator predicting the mean of the training targets."""

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean = np.mean(y)
        else:
            self.mean = np.average(y, weights=sample_weight)

    def predict(self, X):
        check_is_fitted(self, 'mean')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.mean)
        return y


class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.

        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray, shape=(n, m)
            The data array.
        y : ndarray, shape=(n,)
            The target labels.
        residual : ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : ndarray, shape=(n,)
            The predictions.
        sample_weight : ndarray, shape=(n,)
            The weight of each sample.
        sample_mask : ndarray, shape=(n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.

        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Template method for updating terminal regions (=leaves). """


class RegressionLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for regression loss functions. """

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(RegressionLossFunction, self).__init__(n_classes)


class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares. """

    def init_estimator(self):
        return MeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            return np.mean((y - pred.ravel()) ** 2.0)
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * ((y - pred.ravel()) ** 2.0)))

    def negative_gradient(self, y, pred, **kargs):
        return y - pred.ravel()

    def negative_gradient_i(self, i, y, train_scores, **kargs):

        train_scores_sum = np.sum(train_scores, axis=1)
        train_scores_sum -= train_scores[:, i]
        return y - train_scores_sum

        # train_scores_sum = np.sum(train_scores, axis=1)
        # train_scores_sum -= train_scores[:, i]
        # train_scores_sum /= train_scores.shape[1]
        # return y - train_scores_sum

    def update_terminal_regions_un(self, tree, X, y, residual, y_pred,
                                   sample_weight, sample_mask,
                                   learning_rate=1.0, k=0, criterion='mseprob'):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        if criterion == 'mseprob':
            F = [f for f in tree.feature if f != -2]
            for s_current_node in range(len(F)):
                for kk in range(s_current_node + 1, len(F)):
                    if F[s_current_node] == F[kk]:
                        F[kk] = -1
            F = np.array(F)
            y_pred[:, k] += learning_rate * tree.predict3(X, F).ravel()
        else:
            y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def update_terminal_regions_un_i(self, tree, i, X, y, residual, train_scores,
                                     sample_weight, sample_mask,
                                     learning_rate=1.0, k=0, criterion='mseprob'):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        if tree.node_count == 1:
            train_scores[:, i].fill(tree.gmma[0])
            # train_scores[:, i].fill(np.mean(y) / train_scores.shape[1])
            return

        # update predictions
        F = [f for f in tree.feature if f != -2]
        for s_current_node in range(len(F)):
            for kk in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[kk]:
                    F[kk] = -1
        F = np.array(F)
        test = tree.predict3(X, F).ravel()
        train_scores[:, i] = test

    def update_terminal_regions_bart(self, tree, i, estimator_region, train_scores):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        if tree.node_count == 1:
            train_scores[:, i].fill(tree.gmma[0])
            return

        sorted_reg = np.argsort(np.argsort(estimator_region))
        organized_gamma = np.zeros(len(tree.gmma))
        for ind, region_id in enumerate(sorted_reg):
            organized_gamma[ind] = tree.gmma[region_id]
        train_scores[:, i] = tree.preg.dot(organized_gamma)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass


LOSS_FUNCTIONS = {'ls': LeastSquaresError}
INIT_ESTIMATORS = {'zero': ZeroEstimator}


class BaseBART(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Abstract base class for Gradient Boosting. """

    @abstractmethod
    def __init__(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features,
                 random_state, alpha=0.95, beta=2, hyper_mu=3, quantile=0.9, verbose=0, max_leaf_nodes=None,
                 n_iteration=100, n_after_burn_iteration=50, warm_start=False, presort='auto', sigma_type='mad',
                 p_prune=0.25, p_grow=0.25, validation_fraction=0.1, n_iter_no_change=None, tol=1e-4, sigma_Xp=None):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        self.hyper_mu = hyper_mu
        self.quantile = quantile
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.n_iteration = n_iteration + n_after_burn_iteration
        self.n_after_burn_iteration = n_after_burn_iteration
        self.warm_start = warm_start
        self.presort = presort
        self.p_prune = p_prune
        self.p_grow = p_grow
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.hyper_lambda = None
        self.posterior_sigma = None
        self.random_seed = random_state
        self.sigma_Xp = sigma_Xp
        self.mh_counter = 0
        self.current_iteration = 0
        self.curr_it_sample = 0
        self.chi_sq_list = None
        self.train_score_ = None
        self.tree_helpers = None
        self.smoothing = True
        self.sigma_mu = (0.5 / (2 * np.sqrt(self.n_estimators))) ** 2
        self.iteration_type_dict = {'P': 0, 'G': 0, 'C': 0}
        self.mh_acceptance = None
        self.p_instances = None
        self.sigma_type = sigma_type

    def _fit_stage(self, i, X, y, sample_weight, sample_mask,
                   random_state, X_idx_sorted, X_csc=None, X_csr=None):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == np.bool
        loss = self.loss_

        new_splitter = 'rbart'

        node_to_prune = -1

        for k in range(loss.K):

            residual = loss.negative_gradient_i(i, y, self.train_score_, k=k,
                                                sample_weight=sample_weight)

            if self.estimators_[i, k] is None:

                # induce regression tree on residuals
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    splitter=new_splitter,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    min_impurity_split=self.min_impurity_split,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=random_state,
                    presort=self.presort,
                    tol=self.sigma_Xp,
                    samples=None)

                first = True
                new_tree = copy.copy(tree)
                node_count = 1
            else:
                first = False
                tree = self.estimators_[i, 0]
                new_tree = copy.copy(tree)
                new_tree.extra_copy(tree, self.sigma_Xp)
                node_count = new_tree.tree_.node_count

            if first:
                iteration_type = 'G'
                temp_feature = self.pick_random_feature(i, X, 0, X.shape[0])
                temp_split_val = self.pick_random_split(i, X, temp_feature, 0, X.shape[0])
                tree_helper = TreeHelper(0, X.shape[0], -1, -1, 0, 0, 0, 0, temp_feature, temp_split_val)
            else:
                iteration_type = self.pick_generation_step(i)
                if iteration_type in ['P', 'C']:
                    node_to_prune = self.pick_random_prune_node(i)

                tree_helper = self._get_a_random_node(i, X, iteration_type, node_to_prune)
                if tree_helper.end == 0:
                    tree_helper.end = X.shape[0]

            can_feature_expand = tree_helper.feature != -1
            self.iteration_type_dict[iteration_type] += 1
            if iteration_type in ['P', 'C']:
                new_tree.prune(residual, node_to_prune, self.estimators_regions[i])
            if iteration_type in ['C', 'G'] and can_feature_expand:
                new_tree = new_tree.expand(X, residual, tree_helper, sample_weight=sample_weight,
                                           check_input=False, X_idx_sorted=X_idx_sorted,
                                           expansion=not first)

            cond_1 = iteration_type in ['G', 'P'] and node_count == new_tree.tree_.node_count
            cond_2 = iteration_type == 'C' and node_count - 2 == new_tree.tree_.node_count
            self.mh_acceptance[self.curr_it_sample, i] = False
            if not (cond_1 or cond_2):
                if iteration_type == 'P':
                    mh_ratio = self.mh_prune(X, new_tree, tree_helper, residual, i)
                elif iteration_type == 'G':
                    mh_ratio = self.mh_grow(X, new_tree, tree_helper, residual, i)
                else:
                    mh_ratio = self.mh_smooth_change(i, new_tree, residual, tree_helper)

                random.seed(self.random_seed)
                self.random_seed += 1
                mh_rand = math.log(random.random())
                # print(self.current_iteration, i, iteration_type, 'mh_ratio', mh_ratio, 'mh_rand', mh_rand)
                if mh_rand < mh_ratio:
                    self.mh_counter += 1
                    self.mh_acceptance[self.curr_it_sample, i] = True
                    """
                    print(self.mh_counter, 'Type:', iteration_type, 'Iteration:', self.current_iteration,
                          'Tree:', i, 'Feature:', tree_helper.feature, 'Threshold:', tree_helper.threshold,
                          'Node Count:', new_tree.tree_.node_count, 'ID', tree_helper.node_id,
                          'Prune node:', node_to_prune, 'N_split Point', self.tree_nb_splits[i], mh_ratio)
                    """
                    tree = new_tree

                    if iteration_type == 'G':
                        self.estimators_regions[i][tree_helper.curr_region] = tree.tree_.node_count - 2
                        self.estimators_regions[i].append(tree.tree_.node_count - 1)
                    elif iteration_type in ['C', 'P']:
                        nodes = self.estimators_[i, 0].tree_.__getstate__()['nodes']
                        right_c = nodes[node_to_prune][1]
                        self.estimators_regions[i][tree_helper.curr_region] = node_to_prune
                        right_curr_region = np.where(np.array(self.estimators_regions[i]) == right_c)[0][0]
                        self.estimators_regions[i] = list(np.delete(self.estimators_regions[i], right_curr_region))
                        for ind, reg in enumerate(self.estimators_regions[i]):
                            if reg > right_c:
                                self.estimators_regions[i][ind] -= 2
                        if iteration_type == 'C':
                            self.estimators_regions[i][tree_helper.curr_region] = tree.tree_.node_count - 2
                            self.estimators_regions[i].append(tree.tree_.node_count - 1)
                else:
                    if first and new_tree.tree_.node_count == 3:
                        new_tree.prune(residual, 0, [1, 2])

            if first and new_tree.tree_.node_count == 1:
                tree = new_tree

            self.estimators_[i, 0] = tree
            new_gmma = self.update_mukj(i, residual)
            tree.update_gmma(new_gmma)

            loss.update_terminal_regions_bart(tree.tree_, i, self.estimators_regions[i], self.train_score_)

            if self.current_iteration >= (self.n_iteration - self.n_after_burn_iteration):
                self.estimators_samplers[self.curr_it_sample, i] = tree

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        loss_class = LOSS_FUNCTIONS[self.loss]

        self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            if isinstance(self.init, six.string_types):
                if self.init not in INIT_ESTIMATORS:
                    raise ValueError('init="%s" is not supported' % self.init)
            else:
                if (not hasattr(self.init, 'fit')
                        or not hasattr(self.init, 'predict')):
                    raise ValueError("init=%r must be valid BaseEstimator "
                                     "and support both fit and "
                                     "predict" % self.init)

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, np.integer, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

        allowed_presort = ('auto', True, False)
        if self.presort not in allowed_presort:
            raise ValueError("'presort' should be in {}. Got {!r} instead."
                             .format(allowed_presort, self.presort))

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        if self.init is None:
            self.init_ = self.loss_.init_estimator()
        elif isinstance(self.init, six.string_types):
            self.init_ = INIT_ESTIMATORS[self.init]()
        else:
            self.init_ = self.init

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)

        self.estimators_samplers = np.empty((self.n_after_burn_iteration, self.n_estimators), dtype=np.object)
        self.mh_acceptance = np.empty((self.n_after_burn_iteration, self.n_estimators), dtype=np.bool)
        self.tree_helpers = np.empty(self.n_estimators, dtype=np.object)
        self.estimators_regions = [[0] for _ in range(self.n_estimators)]
        self.tree_nb_splits = np.ones(self.n_estimators)
        self.p_instances = np.empty(self.n_estimators, dtype=np.object)

        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_.resize((total_n_estimators, self.loss_.K))
        self.train_score_.resize(total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_.resize(total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self, 'estimators_')

    @property
    @deprecated("Attribute n_features was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def n_features(self):
        return self.n_features_

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        sigma_mult: Multiplier for the sigmas for uncertain trees.

        Returns
        -------
        self : object
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)

        check_consistent_length(X, y, sample_weight)

        y = self._validate_y(y, sample_weight)

        if self.n_iter_no_change is not None:
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction))
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        # init state
        self._init_state()

        init_residuals = lr().fit(X, y).predict(X)
        # init_residuals = sm.OLS(y, X).fit().predict(X)
        init_residuals = y - init_residuals
        sample_var_y = np.var(init_residuals)

        random.seed(self.random_seed)
        self.chi_sq_list = chi2.rvs(df=self.hyper_mu + X.shape[0], size=1000, random_state=self.random_seed)
        self.random_seed += 1
        self.calculate_hyper_parameters_squared(sample_var_y)

        temp_alpha = self.hyper_mu / 2
        temp_beta = 2 / (self.hyper_mu * self.hyper_lambda)
        self.posterior_sigma = self._bart_inverse_gama_sampler(alpha=temp_alpha, beta=temp_beta)
        # self.posterior_sigma = 0.05

        # fit initial model - FIXME make sample_weight optional
        self.init_.fit(X, y, sample_weight)

        # init predictions
        #########################################################
        # y_pred = self.init_.predict(X)
        # mean_prediction = y_pred / self.n_estimators
        # mean_prediction = y_pred
        #########################################################
        # self.train_score_ = np.full((X.shape[0], self.n_estimators), mean_prediction, dtype=np.float64)
        self.train_score_ = np.zeros((X.shape[0], self.n_estimators), dtype=np.float64)
        begin_at_stage = 0

        # The rng state must be preserved if warm_start is True
        self._rng = check_random_state(self.random_state)

        if self.presort is True and issparse(X):
            raise ValueError(
                "Presorting is not supported for sparse matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if presort == 'auto':
            presort = not issparse(X)

        X_idx_sorted = None
        if presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        # self.sigma_xp = np.std(X, axis=0, dtype=np.float64)

        # fit the boosting stages
        #########################################################
        if self.sigma_Xp < 0.00000001:
            self.smoothing = False

        if self.sigma_type == 'mad':
            sigma_xp = np.median(np.absolute(X - np.median(X, axis=0)), axis=0)
            # sigma_xp = stats.median_absolute_deviation(X, axis=0)
            sigma_xp = np.array(sigma_xp, dtype=np.float64)
        else:
            sigma_xp = np.std(X, axis=0, dtype=np.float64)

        sigma_xp = np.where(sigma_xp == 0, 1e-15, sigma_xp)
        self.sigma_Xp = sigma_xp * self.sigma_Xp
        #########################################################
        n_stages = self._fit_stages(X, y, sample_weight, self._rng,
                                    X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if hasattr(self, 'oob_improvement_'):
            self.oob_improvement_ = self.oob_improvement_[:n_stages]

        print('mh counter', self.mh_counter)
        print('Iteration Dict', self.iteration_type_dict)
        return self

    def _fit_stages(self, X, y, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.ones(self.n_iter_no_change) * np.inf
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_decision_function(X_val)

        # perform boosting iterations
        i = begin_at_stage
        iteration = begin_at_stage
        for iteration in range(self.n_iteration):
            self.current_iteration = iteration
            if iteration % 5 == 0:
                print('ITERATION', iteration, self.mh_counter)
            for i in range(begin_at_stage, self.n_estimators):
                # fit next stage of trees
                self._fit_stage(i, X, y, sample_weight,
                                sample_mask, random_state, X_idx_sorted,
                                X_csc, X_csr)

                if monitor is not None:
                    early_stopping = monitor(i, self, locals())
                    if early_stopping:
                        break

                # We also provide an early stopping based on the score from
                # validation set (X_val, y_val), if n_iter_no_change is set
                if self.n_iter_no_change is not None:
                    # By calling next(y_val_pred_iter), we get the predictions
                    # for X_val after the addition of the current stage
                    validation_loss = loss_(y_val, next(y_val_pred_iter),
                                            sample_weight_val)

                    # Require validation_score to be better (less) than at least
                    # one of the last n_iter_no_change evaluations
                    if np.any(validation_loss + self.tol < loss_history):
                        loss_history[i % len(loss_history)] = validation_loss
                    else:
                        break

            ##################################################################
            # Update mu and sigma
            alpha = (self.hyper_mu + X.shape[0]) / 2
            error = y - np.sum(self.train_score_, axis=1)
            error = error ** 2
            error_sum = np.sum(error)
            beta = 2 / (error_sum + self.hyper_mu * self.hyper_lambda)
            self.posterior_sigma = self._bart_inverse_gama_sampler(alpha, beta)
            if iteration >= (self.n_iteration - self.n_after_burn_iteration):
                self.curr_it_sample += 1
            # print(self.estimators_regions)
            ##################################################################
        return iteration + 1

    def _init_decision_function(self, X):
        """Check input and compute prediction of ``init``. """
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        score = self.init_.predict(X).astype(np.float64)
        return score

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        score = np.zeros(X.shape[0])
        # current_burn = 0
        if self.criterion == 'mseprob':
            for iter_ind, iter_samples_trees in enumerate(self.estimators_samplers):
                temp_score = 0
                for tree_ind, tree in enumerate(iter_samples_trees):
                    if tree.tree_.node_count == 1:
                        score += tree.tree_.gmma[0]
                    else:
                        if self.p_instances[tree_ind] is None or self.mh_acceptance[iter_ind, tree_ind]:
                            F = [f for f in tree.tree_.feature if f != -2]
                            for s_current_node in range(len(F)):
                                for kk in range(s_current_node + 1, len(F)):
                                    if F[s_current_node] == F[kk]:
                                        F[kk] = -1
                            F = np.array(F)
                            temp_score += tree.predict5(X, F).ravel()
                            self.p_instances[tree_ind] = tree.tree_.preg
                        else:
                            temp_score += self.p_instances[tree_ind][:len(X), :].dot(tree.tree_.gmma)

                # print('Score:', temp_score, current_burn)
                # current_burn += 1
                score += temp_score
            score /= self.n_after_burn_iteration
        else:
            predict_stages(self.estimators_, X, 1, score)
        return score

    def _staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        score = self._init_decision_function(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, score)
            yield score.copy()

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        self._check_initialized()

        total_sum = np.zeros((self.n_features_,), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators, n_classes]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves

    def _get_a_random_node(self, i, X, iteration_type, node_to_prune):

        tree = self.estimators_[i, 0]
        nodes = tree.tree_.__getstate__()['nodes']
        candidates = self.estimators_regions[i]

        if iteration_type == 'G':
            random.seed(self.random_seed)
            self.random_seed += 1
            rand_ind = random.randint(0, len(candidates) - 1)
            node_id = candidates[rand_ind]
            curr_region = np.where(np.array(candidates) == node_id)[0][0]
            curr_node = nodes[node_id]
        else:
            node_id = node_to_prune
            curr_node = nodes[node_id]
            node_lc_id = curr_node[0]
            curr_region = np.where(np.array(candidates) == node_lc_id)[0][0]
        parent_id = -1
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        for p_id, node in enumerate(nodes):
            if node_id in [node[0], node[1]]:
                parent_id = p_id
                break
        node_depth = np.zeros(shape=len(nodes), dtype=np.int)
        start_end_list = np.zeros((len(nodes), 2), dtype=np.int)
        start_end_list[0, 1] = nodes[0][5]
        stack = [(0, -1)]
        while len(stack) > 0:
            nd_id, parent_depth = stack.pop()
            node_depth[nd_id] = parent_depth + 1

            if children_left[nd_id] != children_right[nd_id]:
                stack.append((children_left[nd_id], parent_depth + 1))
                stack.append((children_right[nd_id], parent_depth + 1))
                left_child = nodes[children_left[nd_id]]

                start_end_list[children_left[nd_id]][0] = start_end_list[nd_id][0]

                temp_val = start_end_list[children_left[nd_id]][0] + left_child[5]
                start_end_list[children_left[nd_id]][1] = temp_val
                start_end_list[children_right[nd_id]][0] = temp_val
                start_end_list[children_right[nd_id]][1] = start_end_list[nd_id][1]

        start = start_end_list[node_id][0]
        end = start_end_list[node_id][1]
        depth = node_depth[node_id]
        if parent_id == -1:
            impurity = -1
        else:
            impurity = curr_node[4]
        is_left = node_id == nodes[parent_id][0]

        if iteration_type != 'P':
            temp_feature = self.pick_random_feature(i, X, start, end)
            temp_split_val = self.pick_random_split(i, X, temp_feature, start, end)
        else:
            temp_feature = curr_node[2]
            temp_split_val = self.pick_random_split(i, X, temp_feature, start, end)

        return TreeHelper(start, end, parent_id, impurity, depth, curr_region,
                          node_id, is_left, temp_feature, temp_split_val)

    def _inverse_gama_sampler(self, alpha, beta):
        """Generate a value from an inverse gamma distribution. """
        random.seed(self.random_seed)
        res = invgamma.rvs(a=alpha, scale=beta, size=1, random_state=self.random_seed)[0]
        self.random_seed += 1
        return res

    def _bart_inverse_gama_sampler(self, alpha, beta):
        """Generate a value from an inverse gamma distribution. """
        random.seed(self.random_seed)
        res = (1 / (beta / 2)) / self.chi_sq_list[random.randint(0, len(self.chi_sq_list) - 1)]
        self.random_seed += 1
        return res

    def _normal_distribution_sampler(self, loc, scale):
        """Generate a value from an inverse gamma distribution. """
        random.seed(self.random_seed)
        res = norm.rvs(loc=loc, scale=scale, size=1, random_state=self.random_seed)[0]
        self.random_seed += 1
        return res

    def calculate_hyper_parameters(self, y):

        sample_y_std = np.std(y)
        ten_pctile_chisq_df_hyper_nu = chi2.ppf(q=1 - self.quantile, df=self.hyper_mu)
        self.hyper_lambda = ten_pctile_chisq_df_hyper_nu / self.hyper_mu * sample_y_std

    def calculate_hyper_parameters_squared(self, sample_var_y):

        ten_pctile_chisq_df_hyper_nu = chi2.ppf(q=1 - self.quantile, df=self.hyper_mu)
        self.hyper_lambda = ten_pctile_chisq_df_hyper_nu / self.hyper_mu * sample_var_y

    def grow_transition_ratio(self, X, i, start, end):

        ng_nodes = self.ng_nodes(i)

        start_end_range = list(range(start, end))
        samples = start_end_range
        if self.estimators_[i, 0] is not None:
            samples = self.estimators_[i, 0].tree_.samples[start_end_range]

        np_features = 0
        for ind in range(X.shape[1]):
            if len(set(X[samples, ind])) > 1:
                np_features += 1

        np_splits = self.tree_nb_splits[i]

        npr_nodes = self.npr_nodes(i)
        if npr_nodes == 0:
            npr_nodes = 1

        # return math.log(ng_nodes) + math.log(np_features) + math.log(np_splits) - math.log(npr_nodes)
        grow_prune_ration = math.log(self.p_prune) - math.log(self.p_grow)
        return math.log(ng_nodes) + math.log(np_features) + math.log(np_splits) - math.log(npr_nodes) \
               + grow_prune_ration

    def npr_nodes(self, i):
        if self.estimators_[i, 0] is None:
            return 1
            # return self.p_grow

        prunable_nodes = 0
        nodes = self.estimators_[i, 0].tree_.__getstate__()['nodes']
        for node in nodes:
            if (node[0] in self.estimators_regions[i]) and (node[1] in self.estimators_regions[i]):
                prunable_nodes += 1
        if prunable_nodes == 0:
            return self.p_prune
        return prunable_nodes

    def prune_transition_ratio(self, X, i, start, end):

        ng_nodes = self.ng_nodes(i)
        start_end_range = list(range(start, end))
        samples = start_end_range
        if self.estimators_[i, 0] is not None:
            samples = self.estimators_[i, 0].tree_.samples[start_end_range]

        np_features = 0
        for ind in range(X.shape[1]):
            if len(set(X[samples, ind])) > 1:
                np_features += 1
        np_splits = self.tree_nb_splits[i]

        npr_nodes = self.npr_nodes(i)
        if npr_nodes == 0:
            npr_nodes = 1

        # return math.log(npr_nodes) - math.log(ng_nodes - 1) - math.log(np_features) - math.log(np_splits)
        grow_prune_ration = math.log(self.p_grow) - math.log(self.p_prune)
        return math.log(npr_nodes) + grow_prune_ration \
               - math.log(ng_nodes - 1) - math.log(np_features) - math.log(np_splits)

    def ng_nodes(self, i):
        return len(self.estimators_regions[i])

    def pick_random_prune_node(self, i):
        if self.estimators_[i, 0] is None:
            return -1
        prunable_nodes = []
        nodes = self.estimators_[i, 0].tree_.__getstate__()['nodes']
        for ind, node in enumerate(nodes):
            if node[0] in self.estimators_regions[i] and node[1] in self.estimators_regions[i]:
                prunable_nodes.append(ind)

        random.seed(self.random_seed)
        rand_ind = random.randint(0, len(prunable_nodes) - 1)
        self.random_seed += 1
        return prunable_nodes[rand_ind]

    def pick_random_feature(self, i, X, start, end):

        features = np.arange(X.shape[1])
        iteration = 0
        while True and iteration < 100:
            random.seed(self.random_seed)
            f = random.randint(0, len(features) - 1)
            # New
            if self.estimators_[i, 0] is None:
                f_values = set(X[:, f])
            else:
                start_end_range = list(range(start, end))
                samples = self.estimators_[i, 0].tree_.samples[start_end_range]
                f_values = set(X[samples, f])

            self.random_seed += 1
            if len(f_values) > 1:
                return f
            else:
                iteration += 1
        return -1

    def pick_random_split(self, i, X, feature, start, end):

        # New
        if self.estimators_[i, 0] is None:
            f_values = sorted(set(X[:, feature]))
        else:
            start_end_range = list(range(start, end))
            samples = self.estimators_[i, 0].tree_.samples[start_end_range]
            f_values = sorted(set(X[samples, feature]))

        # Removing the max value
        self.tree_nb_splits[i] = len(f_values) - 1
        ###########################################################
        random.seed(self.random_seed)
        random_ind = random.randint(0, len(f_values) - 1)
        self.random_seed += 1

        if random_ind < len(f_values) - 1:
            return (f_values[random_ind] + f_values[random_ind + 1]) / 2

        return (f_values[random_ind] + f_values[random_ind - 1]) / 2

    def pick_generation_step(self, i):

        node_count = self.estimators_[i, 0].tree_.node_count

        if node_count == 1:
            return 'G'

        random.seed(self.random_seed)
        rand_val = random.random()
        self.random_seed += 1

        # Grow a node
        if rand_val <= 2.5 / 9:
            return 'G'

        # Make a pruning
        if rand_val <= 5 / 9:
            return 'P'

        # Change the splitting criteria of a node
        return 'C'

    def smooth_grow_likelihood_ratio(self, i, new_tree, tree_helper, residuals):

        old_tree = self.estimators_[i, 0]
        lc_region_id = tree_helper.curr_region
        rc_region_id = len(new_tree.tree_.gmma) - 1

        a = self.region_smooth_likelihood(new_tree, residuals, lc_region_id)
        b = 0
        if not self.smoothing:
            b = self.region_smooth_likelihood(new_tree, residuals, rc_region_id)
        c = self.region_smooth_likelihood(old_tree, residuals, lc_region_id)

        # print('a', a, 'b', b, 'c', c)
        return a + b - c

    def smooth_prune_likelihood_ratio(self, i, new_tree, tree_helper, residuals):

        old_tree = self.estimators_[i, 0]
        old_parent = old_tree.tree_.__getstate__()['nodes'][tree_helper.node_id]
        lc_region_id = tree_helper.curr_region
        rc_region_id = np.where(np.array(self.estimators_regions[i]) == old_parent[1])[0][0]

        a = self.region_smooth_likelihood(old_tree, residuals, lc_region_id)
        b = 0
        if not self.smoothing:
            b = self.region_smooth_likelihood(old_tree, residuals, rc_region_id)
        c = self.region_smooth_likelihood(new_tree, residuals, lc_region_id)
        return a + b - c

    def tree_structure_ratio(self, i, X, tree_depth, start, end):

        start_end_range = list(range(start, end))
        samples = start_end_range
        if self.estimators_[i, 0] is not None:
            samples = self.estimators_[i, 0].tree_.samples[start_end_range]

        np_features = 0
        for ind in range(X.shape[1]):
            if len(set(X[samples, ind])) > 1:
                np_features += 1

        np_splits = self.tree_nb_splits[i]

        # BartMachine
        nominator = math.log(self.alpha) + 2 * math.log(1 - self.alpha / math.pow(2 + tree_depth, self.beta))
        denominator = math.log(math.pow(1 + tree_depth, self.beta) - self.alpha) + \
                      math.log(np_features) + math.log(np_splits)

        return nominator - denominator

    def mh_grow(self, X, new_tree, tree_helper, residuals, i):

        grow_transition_ration = self.grow_transition_ratio(X, i, tree_helper.start, tree_helper.end)
        likelihood_ratio = self.smooth_grow_likelihood_ratio(i, new_tree, tree_helper, residuals)
        tree_structure_ratio = self.tree_structure_ratio(i, X, tree_helper.depth, tree_helper.start, tree_helper.end)
        mh = grow_transition_ration + likelihood_ratio + tree_structure_ratio  # + (0.5 * math.log(2 * math.pi))
        return mh

    def mh_prune(self, X, new_tree, tree_helper, residuals, i):

        prune_transition_ration = self.prune_transition_ratio(X, i, tree_helper.start, tree_helper.end)
        likelihood_ratio = self.smooth_prune_likelihood_ratio(i, new_tree, tree_helper, residuals)
        tree_structure_ratio = self.tree_structure_ratio(i, X, tree_helper.depth, tree_helper.start, tree_helper.end)
        mh = prune_transition_ration - likelihood_ratio - tree_structure_ratio  # - (0.5 * math.log(2 * math.pi))
        return mh

    def mh_smooth_change(self, i, new_tree, residuals, tree_helper):

        lc_region_id = tree_helper.curr_region
        rc_region_id = len(new_tree.tree_.gmma) - 1

        old_tree = self.estimators_[i, 0]
        old_nodes = old_tree.tree_.__getstate__()['nodes']
        old_lc_region_id = tree_helper.curr_region
        old_parent = old_nodes[tree_helper.node_id]
        old_rc_region_id = np.where(np.array(self.estimators_regions[i]) == old_parent[1])[0][0]

        b = 0
        d = 0
        a = self.region_smooth_likelihood(new_tree, residuals, lc_region_id)
        c = self.region_smooth_likelihood(old_tree, residuals, old_lc_region_id)
        if not self.smoothing:
            b = self.region_smooth_likelihood(new_tree, residuals, rc_region_id)
            d = self.region_smooth_likelihood(old_tree, residuals, old_rc_region_id)

        return a + b - c - d

    def region_smooth_likelihood(self, tree, residuals, regions_id):

        n_observations = len(residuals)
        sigma_0_inv, temp_det = self.calculate_likelihood_matrix(tree, n_observations, regions_id)
        """
        try:
            sigma_0 = np.linalg.inv(sigma_0_inv)
        except np.linalg.LinAlgError as _:
            sigma_0 = np.linalg.pinv(sigma_0_inv)

        sigma_det = np.linalg.slogdet(sigma_0)
        b = 0
        if sigma_det[0] != 0:
            sigma_det = sigma_det[0] * sigma_det[1]
            b = -0.5 * sigma_det
        """
        """
        sigma_det = np.linalg.slogdet(sigma_0_inv)
        b = 0
        if sigma_det[0] != 0:
            sigma_det = sigma_det[0] * sigma_det[1]
            b = 0.5 * sigma_det
        """
        # b = -0.5 * temp_det
        b = 0.5 * temp_det

        if self.smoothing:
            c = -0.5 * residuals.T.dot(sigma_0_inv).dot(residuals)
        else:
            if tree is None:
                preg = np.ones(n_observations).reshape(n_observations, 1)
            else:
                preg = tree.tree_.preg
            region_preg = preg[:, regions_id]
            region_idx = np.where(region_preg > 0)[0]
            region_residuals = residuals[region_idx]
            c = -0.5 * region_residuals.T.dot(sigma_0_inv).dot(region_residuals)

        return b + c

    def update_mukj(self, i, residuals):

        tree = self.estimators_[i, 0]
        new_gmma = np.zeros(len(self.estimators_regions[i]))
        preg = tree.tree_.preg
        gammas = np.linalg.pinv(preg.T.dot(preg)).dot(preg.T).dot(residuals)
        est_reg_copy = np.copy(self.estimators_regions[i])
        sorted_reg = np.argsort(est_reg_copy)

        for ind, region_id in enumerate(gammas):

            region_preg = preg[:, ind]
            region_preg_sq = region_preg ** 2
            sum_prob = np.sum(region_preg_sq)
            sum_prediction = np.zeros(preg.shape[0])

            for ind_bar, gamma_bar in enumerate(gammas):
                if ind_bar == ind:
                    continue
                region_bar_preg = preg[:, ind_bar]
                sum_prediction += region_bar_preg * gamma_bar

            sum_prediction = region_preg * (residuals - sum_prediction)
            sum_prediction = np.sum(sum_prediction)
            mean_nominator = self.sigma_mu * sum_prediction

            denominator = sum_prob * self.sigma_mu + self.posterior_sigma
            var_nominator = self.posterior_sigma * self.sigma_mu
            variance = var_nominator / denominator
            mean = mean_nominator / denominator
            random.seed(self.random_seed)
            std = np.sqrt(variance) * random.random()
            self.random_seed += 1
            new_gmma[ind] = self._normal_distribution_sampler(mean, std)

        """
        if self.current_iteration >= 6 and i == 7:
            new_gmma[1] = gammas[1]
        """

        real_gmma = np.zeros(len(self.estimators_regions[i]))
        for ind, region_id in enumerate(sorted_reg):
            real_gmma[ind] = new_gmma[region_id]

        """
        if i == 7:
            print(self.current_iteration, i, gammas)
            print(self.current_iteration, i, new_gmma)
            print()
        """

        return real_gmma

    def calculate_likelihood_matrix(self, tree, n_observations, regions_id):

        if tree is None:
            preg = np.ones(n_observations).reshape(n_observations, 1)
        else:
            preg = tree.tree_.preg

        region_preg = preg[:, regions_id]
        region_preg_idx = None
        if self.smoothing:
            sigma_k1_l_inv = self.posterior_sigma * np.identity(n_observations)
        else:
            region_preg_idx = np.where(region_preg > 0)[0]
            sigma_k1_l_inv = self.posterior_sigma * np.identity(len(region_preg_idx))

        np.fill_diagonal(sigma_k1_l_inv, 1 / sigma_k1_l_inv[0, 0])
        denom_second_term = 1 / self.sigma_mu

        # temp_det = 0
        for t_region in range(preg.shape[1]):

            if not self.smoothing and t_region != regions_id:
                continue

            if self.smoothing:
                t_region_preg = preg[:, t_region]
                c_i_k1_l = t_region_preg
                c_i_k1_l = c_i_k1_l.reshape(len(c_i_k1_l), 1)
            else:
                c_i_k1_l = region_preg[region_preg_idx]
                c_i_k1_l = c_i_k1_l.reshape(len(c_i_k1_l), 1)

            denominator = c_i_k1_l.T.dot(sigma_k1_l_inv).dot(c_i_k1_l)[0][0] + denom_second_term
            # temp_det += math.log(denominator)
            nominator = sigma_k1_l_inv.dot(c_i_k1_l).dot(c_i_k1_l.T).dot(sigma_k1_l_inv)
            division = nominator / denominator
            sigma_k1_l_inv = sigma_k1_l_inv - division
        """
        if self.smoothing:
            temp_det += preg.shape[1] * math.log(self.sigma_mu) + (n_observations * math.log(self.posterior_sigma))
        else:
            temp_det += math.log(self.sigma_mu) + (len(region_preg_idx) * math.log(self.posterior_sigma))
        """
        temp_det = np.linalg.slogdet(sigma_k1_l_inv)
        return sigma_k1_l_inv, temp_det[0] * temp_det[1]
        # return sigma_k1_l_inv, temp_det


class BARTRegressor(BaseBART, RegressorMixin):
    """Gradient Boosting for regression."""

    _SUPPORTED_LOSS = 'ls'

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=100, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None, sigma_type='mad',
                 max_features=None, alpha=0.95, beta=2, hyper_mu=3, quantile=0.9, verbose=0, max_leaf_nodes=None,
                 n_iteration=50, n_after_burn_iteration=50, warm_start=False, presort='auto', p_prune=0.25, p_grow=0.25,
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, sigma_Xp=None):
        super(BARTRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, beta=beta, hyper_mu=hyper_mu, quantile=quantile, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, n_iteration=n_iteration, warm_start=warm_start,
            n_after_burn_iteration=n_after_burn_iteration, sigma_type=sigma_type,
            presort=presort, p_prune=p_prune, p_grow=p_grow, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, sigma_Xp=sigma_Xp)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        return self._decision_function(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for y in self._staged_decision_function(X):
            yield y.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super(BARTRegressor, self).apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves
