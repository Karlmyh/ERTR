from __future__ import division

import pytest

import numpy as np
from scipy import sparse

from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_false

from sklearn.impute import SimpleImputer, MICEImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.random_projection import sparse_random_matrix


def _check_statistics(X, X_true,
                      strategy, statistics, missing_values):
    """Utility function for testing imputation for a given strategy.

    Test:
        - along the two axes
        - with dense and sparse arrays

    Check that:
        - the statistics (mean, median, mode) are correct
        - the missing values are imputed correctly"""

    err_msg = "Parameters: strategy = %s, missing_values = %s, " \
              "axis = {0}, sparse = {1}" % (strategy, missing_values)

    assert_ae = assert_array_equal
    if X.dtype.kind == 'f' or X_true.dtype.kind == 'f':
        assert_ae = assert_array_almost_equal

    # Normal matrix
    imputer = SimpleImputer(missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics,
              err_msg=err_msg.format(0, False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(0, False))

    # Sparse matrix
    imputer = SimpleImputer(missing_values, strategy=strategy)
    imputer.fit(sparse.csc_matrix(X))
    X_trans = imputer.transform(sparse.csc_matrix(X.copy()))

    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    assert_ae(imputer.statistics_, statistics,
              err_msg=err_msg.format(0, True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(0, True))


def test_imputation_shape():
    # Verify the shapes of the imputed matrix for different strategies.
    X = np.random.randn(10, 2)
    X[::2] = np.nan

    for strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = imputer.fit_transform(sparse.csr_matrix(X))
        assert X_imputed.shape == (10, 2)
        X_imputed = imputer.fit_transform(X)
        assert X_imputed.shape == (10, 2)

        mice_imputer = MICEImputer(initial_strategy=strategy)
        X_imputed = mice_imputer.fit_transform(X)
        assert X_imputed.shape == (10, 2)


def safe_median(arr, *args, **kwargs):
    # np.median([]) raises a TypeError for numpy >= 1.10.1
    length = arr.size if hasattr(arr, 'size') else len(arr)
    return np.nan if length == 0 else np.median(arr, *args, **kwargs)


def safe_mean(arr, *args, **kwargs):
    # np.mean([]) raises a RuntimeWarning for numpy >= 1.10.1
    length = arr.size if hasattr(arr, 'size') else len(arr)
    return np.nan if length == 0 else np.mean(arr, *args, **kwargs)


def test_imputation_mean_median():
    # Test imputation using the mean and median strategies, when
    # missing_values != 0.
    rng = np.random.RandomState(0)

    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)

    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = - values[4::2]

    tests = [("mean", "NaN", lambda z, v, p: safe_mean(np.hstack((z, v)))),
             ("mean", 0, lambda z, v, p: np.mean(v)),
             ("median", "NaN", lambda z, v, p: safe_median(np.hstack((z, v)))),
             ("median", 0, lambda z, v, p: np.median(v))]

    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])

        # Create a matrix X with columns
        #    - with only zeros,
        #    - with only missing values
        #    - with zeros, missing values and values
        # And a matrix X_true containing all true values
        for j in range(shape[1]):
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec
                                    - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values

            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]

            true_statistics[j] = true_value_fun(z, v, p)

            # Create the columns
            X[:, j] = np.hstack((v, z, p))

            if 0 == test_missing_values:
                X_true[:, j] = np.hstack((v,
                                          np.repeat(
                                              true_statistics[j],
                                              nb_missing_values + nb_zeros)))
            else:
                X_true[:, j] = np.hstack((v,
                                          z,
                                          np.repeat(true_statistics[j],
                                                    nb_missing_values)))

            # Shuffle them the same way
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])

        # Mean doesn't support columns containing NaNs, median does
        if strategy == "median":
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)

        X_true = X_true[:, cols_to_keep]

        _check_statistics(X, X_true, strategy,
                          true_statistics, test_missing_values)


def test_imputation_median_special_cases():
    # Test median imputation with sparse boundary cases
    X = np.array([
        [0, np.nan, np.nan],  # odd: implicit zero
        [5, np.nan, np.nan],  # odd: explicit nonzero
        [0, 0, np.nan],    # even: average two zeros
        [-5, 0, np.nan],   # even: avg zero and neg
        [0, 5, np.nan],    # even: avg zero and pos
        [4, 5, np.nan],    # even: avg nonzeros
        [-4, -5, np.nan],  # even: avg negatives
        [-1, 2, np.nan],   # even: crossing neg and pos
    ]).transpose()

    X_imputed_median = np.array([
        [0, 0, 0],
        [5, 5, 5],
        [0, 0, 0],
        [-5, 0, -2.5],
        [0, 5, 2.5],
        [4, 5, 4.5],
        [-4, -5, -4.5],
        [-1, 2, .5],
    ]).transpose()
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, .5]

    _check_statistics(X, X_imputed_median, "median",
                      statistics_median, 'NaN')


def test_imputation_most_frequent():
    # Test imputation using the most-frequent strategy.
    X = np.array([
        [-1, -1, 0, 5],
        [-1, 2, -1, 3],
        [-1, 1, 3, -1],
        [-1, 2, 3, 7],
    ])

    X_true = np.array([
        [2, 0, 5],
        [2, 3, 3],
        [1, 3, 3],
        [2, 3, 7],
    ])

    # scipy.stats.mode, used in SimpleImputer, doesn't return the first most
    # frequent as promised in the doc but the lowest most frequent. When this
    # test will fail after an update of scipy, SimpleImputer will need to be
    # updated to be consistent with the new (correct) behaviour
    _check_statistics(X, X_true, "most_frequent", [np.nan, 2, 3, 3], -1)


def test_imputation_pipeline_grid_search():
    # Test imputation within a pipeline + gridsearch.
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=0)),
                         ('tree', tree.DecisionTreeRegressor(random_state=0))])

    parameters = {
        'imputer__strategy': ["mean", "median", "most_frequent"]
    }

    X = sparse_random_matrix(100, 100, density=0.10)
    Y = sparse_random_matrix(100, 1, density=0.10).toarray()
    gs = GridSearchCV(pipeline, parameters)
    gs.fit(X, Y)


def test_imputation_copy():
    # Test imputation with copy
    X_orig = sparse_random_matrix(5, 5, density=0.75, random_state=0)

    # copy=True, dense => copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_false(np.all(X == Xt))

    # copy=True, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean",
                            copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_false(np.all(X.data == Xt.data))

    # copy=False, dense => no copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_array_almost_equal(X, Xt)

    # copy=False, sparse csc => no copy
    X = X_orig.copy().tocsc()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean",
                            copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_array_almost_equal(X.data, Xt.data)

    # copy=False, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean",
                            copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_false(np.all(X.data == Xt.data))

    # Note: If X is sparse and if missing_values=0, then a (dense) copy of X is
    # made, even if copy=False.


def test_mice_rank_one():
    rng = np.random.RandomState(0)
    d = 100
    A = rng.rand(d, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(d, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    imputer = MICEImputer(n_imputations=5,
                          n_burn_in=5,
                          verbose=True,
                          random_state=rng)
    X_filled = imputer.fit_transform(X_missing)
    assert_allclose(X_filled, X, atol=0.001)


@pytest.mark.parametrize(
    "imputation_order",
    ['random', 'roman', 'ascending', 'descending', 'arabic']
)
def test_mice_imputation_order(imputation_order):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    X[:, 0] = 1  # this column should not be discarded by MICEImputer

    imputer = MICEImputer(missing_values=0,
                          n_imputations=1,
                          n_burn_in=1,
                          n_nearest_features=5,
                          min_value=0,
                          max_value=1,
                          verbose=False,
                          imputation_order=imputation_order,
                          random_state=rng)
    imputer.fit_transform(X)
    ordered_idx = [i.feat_idx for i in imputer.imputation_sequence_]
    if imputation_order == 'roman':
        assert np.all(ordered_idx[:d-1] == np.arange(1, d))
    elif imputation_order == 'arabic':
        assert np.all(ordered_idx[:d-1] == np.arange(d-1, 0, -1))
    elif imputation_order == 'random':
        ordered_idx_round_1 = ordered_idx[:d-1]
        ordered_idx_round_2 = ordered_idx[d-1:]
        assert ordered_idx_round_1 != ordered_idx_round_2
    elif 'ending' in imputation_order:
        assert len(ordered_idx) == 2 * (d - 1)


@pytest.mark.parametrize(
    "predictor",
    [DummyRegressor(), BayesianRidge(), ARDRegression()]
)
def test_mice_predictors(predictor):
    rng = np.random.RandomState(0)

    n = 100
    d = 10
    X = sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    imputer = MICEImputer(missing_values=0,
                          n_imputations=1,
                          n_burn_in=1,
                          predictor=predictor,
                          random_state=rng)
    imputer.fit_transform(X)

    # check that types are correct for predictors
    hashes = []
    for triplet in imputer.imputation_sequence_:
        assert triplet.predictor
        hashes.append(id(triplet.predictor))

    # check that each predictor is unique
    assert len(set(hashes)) == len(hashes)


def test_mice_clip():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = sparse_random_matrix(n, d, density=0.10,
                             random_state=rng).toarray()

    imputer = MICEImputer(missing_values=0,
                          n_imputations=1,
                          n_burn_in=1,
                          min_value=0.1,
                          max_value=0.2,
                          random_state=rng)

    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])


@pytest.mark.parametrize(
    "strategy",
    ["mean", "median", "most_frequent"]
)
def test_mice_missing_at_transform(strategy):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X_train = rng.randint(low=0, high=3, size=(n, d))
    X_test = rng.randint(low=0, high=3, size=(n, d))

    X_train[:, 0] = 1  # definitely no missing values in 0th column
    X_test[0, 0] = 0  # definitely missing value in 0th column

    mice = MICEImputer(missing_values=0,
                       n_imputations=1,
                       n_burn_in=1,
                       initial_strategy=strategy,
                       random_state=rng).fit(X_train)
    initial_imputer = SimpleImputer(missing_values=0,
                                    strategy=strategy).fit(X_train)

    # if there were no missing values at time of fit, then mice will
    # only use the initial imputer for that feature at transform
    assert np.all(mice.transform(X_test)[:, 0] ==
                  initial_imputer.transform(X_test)[:, 0])


def test_mice_transform_stochasticity():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = sparse_random_matrix(n, d, density=0.10,
                             random_state=rng).toarray()

    imputer = MICEImputer(missing_values=0,
                          n_imputations=1,
                          n_burn_in=1,
                          random_state=rng)
    imputer.fit(X)

    X_fitted_1 = imputer.transform(X)
    X_fitted_2 = imputer.transform(X)

    # sufficient to assert that the means are not the same
    assert np.mean(X_fitted_1) != pytest.approx(np.mean(X_fitted_2))


def test_mice_no_missing():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 100)
    X[:, 0] = np.nan
    m1 = MICEImputer(n_imputations=10, random_state=rng)
    m2 = MICEImputer(n_imputations=10, random_state=rng)
    pred1 = m1.fit(X).transform(X)
    pred2 = m2.fit_transform(X)
    # should exclude the first column entirely
    assert_allclose(X[:, 1:], pred1)
    # fit and fit_transform should both be identical
    assert_allclose(pred1, pred2)


@pytest.mark.parametrize(
    "rank",
    [3, 5]
)
def test_mice_transform_recovery(rank):
    rng = np.random.RandomState(0)
    n = 100
    d = 100
    A = rng.rand(n, rank)
    B = rng.rand(rank, d)
    X_filled = np.dot(A, B)
    # half is randomly missing
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data in half
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = MICEImputer(n_imputations=10,
                          n_burn_in=10,
                          verbose=True,
                          random_state=rng).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, rtol=1e-5, atol=0.1)


def test_mice_additive_matrix():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    A = rng.randn(n, d)
    B = rng.randn(n, d)
    X_filled = np.zeros(A.shape)
    for i in range(d):
        for j in range(d):
            X_filled[:, (i+j) % d] += (A[:, i] + B[:, j]) / 2
    # a quarter is randomly missing
    nan_mask = rng.rand(n, d) < 0.25
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = MICEImputer(n_imputations=25,
                          n_burn_in=10,
                          verbose=True,
                          random_state=rng).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, atol=0.01)
