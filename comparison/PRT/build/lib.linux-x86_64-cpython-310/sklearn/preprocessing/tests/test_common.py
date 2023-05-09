import pytest
import numpy as np

from scipy import sparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.base import clone

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import quantile_transform

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

iris = load_iris()


def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    return X[:, [col]][~np.isnan(X[:, col])]


@pytest.mark.parametrize(
    "est, func, support_sparse",
    [(MinMaxScaler(), minmax_scale, False),
     (QuantileTransformer(n_quantiles=10), quantile_transform, True)]
)
def test_missing_value_handling(est, func, support_sparse):
    # check that the preprocessing method let pass nan
    rng = np.random.RandomState(42)
    X = iris.data.copy()
    n_missing = 50
    X[rng.randint(X.shape[0], size=n_missing),
      rng.randint(X.shape[1], size=n_missing)] = np.nan
    X_train, X_test = train_test_split(X, random_state=1)
    # sanity check
    assert not np.all(np.isnan(X_train), axis=0).any()
    assert np.any(np.isnan(X_train), axis=0).all()
    assert np.any(np.isnan(X_test), axis=0).all()
    X_test[:, 0] = np.nan  # make sure this boundary case is tested

    Xt = est.fit(X_train).transform(X_test)
    # missing values should still be missing, and only them
    assert_array_equal(np.isnan(Xt), np.isnan(X_test))

    # check that the function leads to the same results as the class
    Xt_class = est.transform(X_train)
    Xt_func = func(X_train, **est.get_params())
    assert_array_equal(np.isnan(Xt_func), np.isnan(Xt_class))
    assert_allclose(Xt_func[~np.isnan(Xt_func)], Xt_class[~np.isnan(Xt_class)])

    # check that the inverse transform keep NaN
    Xt_inv = est.inverse_transform(Xt)
    assert_array_equal(np.isnan(Xt_inv), np.isnan(X_test))
    # FIXME: we can introduce equal_nan=True in recent version of numpy.
    # For the moment which just check that non-NaN values are almost equal.
    assert_allclose(Xt_inv[~np.isnan(Xt_inv)], X_test[~np.isnan(X_test)])

    for i in range(X.shape[1]):
        # train only on non-NaN
        est.fit(_get_valid_samples_by_column(X_train, i))
        # check transforming with NaN works even when training without NaN
        Xt_col = est.transform(X_test[:, [i]])
        assert_array_equal(Xt_col, Xt[:, [i]])
        # check non-NaN is handled as before - the 1st column is all nan
        if not np.isnan(X_test[:, i]).all():
            Xt_col_nonan = est.transform(
                _get_valid_samples_by_column(X_test, i))
            assert_array_equal(Xt_col_nonan,
                               Xt_col[~np.isnan(Xt_col.squeeze())])

    if support_sparse:
        est_dense = clone(est)
        est_sparse = clone(est)

        Xt_dense = est_dense.fit(X_train).transform(X_test)
        Xt_inv_dense = est_dense.inverse_transform(Xt_dense)
        for sparse_constructor in (sparse.csr_matrix, sparse.csc_matrix,
                                   sparse.bsr_matrix, sparse.coo_matrix,
                                   sparse.dia_matrix, sparse.dok_matrix,
                                   sparse.lil_matrix):
            # check that the dense and sparse inputs lead to the same results
            Xt_sparse = (est_sparse.fit(sparse_constructor(X_train))
                         .transform(sparse_constructor(X_test)))
            assert_allclose(Xt_sparse.A, Xt_dense)
            Xt_inv_sparse = est_sparse.inverse_transform(Xt_sparse)
            assert_allclose(Xt_inv_sparse.A, Xt_inv_dense)
