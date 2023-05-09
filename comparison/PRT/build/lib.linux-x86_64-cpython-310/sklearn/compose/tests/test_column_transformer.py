"""
Test the ColumnTransformer.
"""

import numpy as np
from scipy import sparse
import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_dict_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose_dense_sparse

from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_extraction import DictVectorizer


class Trans(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 1D Series -> 2D DataFrame
        if hasattr(X, 'to_frame'):
            return X.to_frame()
        # 1D array -> 2D array
        if X.ndim == 1:
            return np.atleast_2d(X).T
        return X


class SparseMatrixTrans(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = len(X)
        return sparse.eye(n_samples, n_samples).tocsr()


def test_column_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    X_res_first1D = np.array([0, 1, 2])
    X_res_second1D = np.array([2, 4, 6])
    X_res_first = X_res_first1D.reshape(-1, 1)
    X_res_both = X_array

    cases = [
        # single column 1D / 2D
        (0, X_res_first),
        ([0], X_res_first),
        # list-like
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # slice
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),
        # boolean mask
        (np.array([True, False]), X_res_first),
    ]

    for selection, res in cases:
        ct = ColumnTransformer([('trans', Trans(), selection)],
                               remainder='drop')
        assert_array_equal(ct.fit_transform(X_array), res)
        assert_array_equal(ct.fit(X_array).transform(X_array), res)

    ct = ColumnTransformer([('trans1', Trans(), [0]),
                            ('trans2', Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)

    # test with transformer_weights
    transformer_weights = {'trans1': .1, 'trans2': 10}
    both = ColumnTransformer([('trans1', Trans(), [0]),
                              ('trans2', Trans(), [1])],
                             transformer_weights=transformer_weights)
    res = np.vstack([transformer_weights['trans1'] * X_res_first1D,
                     transformer_weights['trans2'] * X_res_second1D]).T
    assert_array_equal(both.fit_transform(X_array), res)
    assert_array_equal(both.fit(X_array).transform(X_array), res)

    both = ColumnTransformer([('trans', Trans(), [0, 1])],
                             transformer_weights={'trans': .1})
    assert_array_equal(both.fit_transform(X_array), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_array).transform(X_array), 0.1 * X_res_both)


def test_column_transformer_dataframe():
    pd = pytest.importorskip('pandas')

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=['first', 'second'])

    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_both = X_array

    cases = [
        # String keys: label based

        # scalar
        ('first', X_res_first),
        # list
        (['first'], X_res_first),
        (['first', 'second'], X_res_both),
        # slice
        (slice('first', 'second'), X_res_both),

        # int keys: positional

        # scalar
        (0, X_res_first),
        # list
        ([0], X_res_first),
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # slice
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),

        # boolean mask
        (np.array([True, False]), X_res_first),
        (pd.Series([True, False], index=['first', 'second']), X_res_first),
    ]

    for selection, res in cases:
        ct = ColumnTransformer([('trans', Trans(), selection)],
                               remainder='drop')
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)

    ct = ColumnTransformer([('trans1', Trans(), ['first']),
                            ('trans2', Trans(), ['second'])])
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)

    ct = ColumnTransformer([('trans1', Trans(), [0]),
                            ('trans2', Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)

    # test with transformer_weights
    transformer_weights = {'trans1': .1, 'trans2': 10}
    both = ColumnTransformer([('trans1', Trans(), ['first']),
                              ('trans2', Trans(), ['second'])],
                             transformer_weights=transformer_weights)
    res = np.vstack([transformer_weights['trans1'] * X_df['first'],
                     transformer_weights['trans2'] * X_df['second']]).T
    assert_array_equal(both.fit_transform(X_df), res)
    assert_array_equal(both.fit(X_df).transform(X_df), res)

    # test multiple columns
    both = ColumnTransformer([('trans', Trans(), ['first', 'second'])],
                             transformer_weights={'trans': .1})
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)

    both = ColumnTransformer([('trans', Trans(), [0, 1])],
                             transformer_weights={'trans': .1})
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)

    # ensure pandas object is passes through

    class TransAssert(BaseEstimator):

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            assert_true(isinstance(X, (pd.DataFrame, pd.Series)))
            if isinstance(X, pd.Series):
                X = X.to_frame()
            return X

    ct = ColumnTransformer([('trans', TransAssert(), 'first')],
                           remainder='drop')
    ct.fit_transform(X_df)
    ct = ColumnTransformer([('trans', TransAssert(), ['first', 'second'])])
    ct.fit_transform(X_df)

    # integer column spec + integer column names -> still use positional
    X_df2 = X_df.copy()
    X_df2.columns = [1, 0]
    ct = ColumnTransformer([('trans', Trans(), 0)], remainder='drop')
    assert_array_equal(ct.fit_transform(X_df), X_res_first)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_first)


def test_column_transformer_sparse_array():
    X_sparse = sparse.eye(3, 2).tocsr()

    # no distinction between 1D and 2D
    X_res_first = X_sparse[:, 0]
    X_res_both = X_sparse

    for col in [0, [0], slice(0, 1)]:
        for remainder, res in [('drop', X_res_first),
                               ('passthrough', X_res_both)]:
            ct = ColumnTransformer([('trans', Trans(), col)],
                                   remainder=remainder)
            assert_true(sparse.issparse(ct.fit_transform(X_sparse)))
            assert_allclose_dense_sparse(ct.fit_transform(X_sparse), res)
            assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse),
                                         res)

    for col in [[0, 1], slice(0, 2)]:
        ct = ColumnTransformer([('trans', Trans(), col)])
        assert_true(sparse.issparse(ct.fit_transform(X_sparse)))
        assert_allclose_dense_sparse(ct.fit_transform(X_sparse), X_res_both)
        assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse),
                                     X_res_both)


def test_column_transformer_sparse_stacking():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    col_trans = ColumnTransformer([('trans1', Trans(), [0]),
                                   ('trans2', SparseMatrixTrans(), 1)])
    col_trans.fit(X_array)
    X_trans = col_trans.transform(X_array)
    assert_true(sparse.issparse(X_trans))
    assert_equal(X_trans.shape, (X_trans.shape[0], X_trans.shape[0] + 1))
    assert_array_equal(X_trans.toarray()[:, 1:], np.eye(X_trans.shape[0]))


def test_column_transformer_error_msg_1D():
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T

    col_trans = ColumnTransformer([('trans', StandardScaler(), 0)])
    assert_raise_message(ValueError, "1D data passed to a transformer",
                         col_trans.fit, X_array)
    assert_raise_message(ValueError, "1D data passed to a transformer",
                         col_trans.fit_transform, X_array)

    class TransRaise(BaseEstimator):

        def fit(self, X, y=None):
            raise ValueError("specific message")

        def transform(self, X, y=None):
            raise ValueError("specific message")

    col_trans = ColumnTransformer([('trans', TransRaise(), 0)])
    for func in [col_trans.fit, col_trans.fit_transform]:
        assert_raise_message(ValueError, "specific message", func, X_array)


def test_2D_transformer_output():

    class TransNo2D(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # if one transformer is dropped, test that name is still correct
    ct = ColumnTransformer([('trans1', 'drop', 0),
                            ('trans2', TransNo2D(), 1)])
    assert_raise_message(ValueError, "the 'trans2' transformer should be 2D",
                         ct.fit_transform, X_array)
    ct.fit(X_array)
    assert_raise_message(ValueError, "the 'trans2' transformer should be 2D",
                         ct.transform, X_array)


def test_2D_transformer_output_pandas():
    pd = pytest.importorskip('pandas')

    class TransNo2D(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=['col1', 'col2'])

    # if one transformer is dropped, test that name is still correct
    ct = ColumnTransformer([('trans1', TransNo2D(), 'col1')])
    assert_raise_message(ValueError, "the 'trans1' transformer should be 2D",
                         ct.fit_transform, X_df)
    ct.fit(X_df)
    assert_raise_message(ValueError, "the 'trans1' transformer should be 2D",
                         ct.transform, X_df)


@pytest.mark.parametrize("remainder", ['drop', 'passthrough'])
def test_column_transformer_invalid_columns(remainder):
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # general invalid
    for col in [1.5, ['string', 1], slice(1, 's'), np.array([1.])]:
        ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder)
        assert_raise_message(ValueError, "No valid specification",
                             ct.fit, X_array)

    # invalid for arrays
    for col in ['string', ['string', 'other'], slice('a', 'b')]:
        ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder)
        assert_raise_message(ValueError, "Specifying the columns",
                             ct.fit, X_array)


def test_column_transformer_invalid_transformer():

    class NoTrans(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    ct = ColumnTransformer([('trans', NoTrans(), [0])])
    assert_raise_message(TypeError, "All estimators should implement fit",
                         ct.fit, X_array)


def test_make_column_transformer():
    scaler = StandardScaler()
    norm = Normalizer()
    ct = make_column_transformer(('first', scaler), (['second'], norm))
    names, transformers, columns = zip(*ct.transformers)
    assert_equal(names, ("standardscaler", "normalizer"))
    assert_equal(transformers, (scaler, norm))
    assert_equal(columns, ('first', ['second']))


def test_make_column_transformer_kwargs():
    scaler = StandardScaler()
    norm = Normalizer()
    ct = make_column_transformer(('first', scaler), (['second'], norm),
                                 n_jobs=3, remainder='drop')
    assert_equal(
        ct.transformers,
        make_column_transformer(('first', scaler),
                                (['second'], norm)).transformers)
    assert_equal(ct.n_jobs, 3)
    assert_equal(ct.remainder, 'drop')
    # invalid keyword parameters should raise an error message
    assert_raise_message(
        TypeError,
        'Unknown keyword arguments: "transformer_weights"',
        make_column_transformer, ('first', scaler), (['second'], norm),
        transformer_weights={'pca': 10, 'Transf': 1}
    )


def test_column_transformer_get_set_params():
    ct = ColumnTransformer([('trans1', StandardScaler(), [0]),
                            ('trans2', StandardScaler(), [1])])

    exp = {'n_jobs': 1,
           'remainder': 'passthrough',
           'trans1': ct.transformers[0][1],
           'trans1__copy': True,
           'trans1__with_mean': True,
           'trans1__with_std': True,
           'trans2': ct.transformers[1][1],
           'trans2__copy': True,
           'trans2__with_mean': True,
           'trans2__with_std': True,
           'transformers': ct.transformers,
           'transformer_weights': None}

    assert_dict_equal(ct.get_params(), exp)

    ct.set_params(trans1__with_mean=False)
    assert_false(ct.get_params()['trans1__with_mean'])

    ct.set_params(trans1='passthrough')
    exp = {'n_jobs': 1,
           'remainder': 'passthrough',
           'trans1': 'passthrough',
           'trans2': ct.transformers[1][1],
           'trans2__copy': True,
           'trans2__with_mean': True,
           'trans2__with_std': True,
           'transformers': ct.transformers,
           'transformer_weights': None}

    assert_dict_equal(ct.get_params(), exp)


def test_column_transformer_named_estimators():
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T
    ct = ColumnTransformer([('trans1', StandardScaler(), [0]),
                            ('trans2', StandardScaler(with_std=False), [1])])
    assert_false(hasattr(ct, 'transformers_'))
    ct.fit(X_array)
    assert_true(hasattr(ct, 'transformers_'))
    assert_true(isinstance(ct.named_transformers_['trans1'], StandardScaler))
    assert_true(isinstance(ct.named_transformers_.trans1, StandardScaler))
    assert_true(isinstance(ct.named_transformers_['trans2'], StandardScaler))
    assert_true(isinstance(ct.named_transformers_.trans2, StandardScaler))
    assert_false(ct.named_transformers_.trans2.with_std)
    # check it are fitted transformers
    assert_equal(ct.named_transformers_.trans1.mean_, 1.)


def test_column_transformer_cloning():
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T

    ct = ColumnTransformer([('trans', StandardScaler(), [0])])
    ct.fit(X_array)
    assert_false(hasattr(ct.transformers[0][1], 'mean_'))
    assert_true(hasattr(ct.transformers_[0][1], 'mean_'))

    ct = ColumnTransformer([('trans', StandardScaler(), [0])])
    ct.fit_transform(X_array)
    assert_false(hasattr(ct.transformers[0][1], 'mean_'))
    assert_true(hasattr(ct.transformers_[0][1], 'mean_'))


def test_column_transformer_get_feature_names():
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T
    ct = ColumnTransformer([('trans', Trans(), [0, 1])])
    # raise correct error when not fitted
    assert_raises(NotFittedError, ct.get_feature_names)
    # raise correct error when no feature names are available
    ct.fit(X_array)
    assert_raise_message(AttributeError,
                         "Transformer trans (type Trans) does not provide "
                         "get_feature_names", ct.get_feature_names)

    # working example
    X = np.array([[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],
                  [{'c': 5}, {'c': 6}]], dtype=object).T
    ct = ColumnTransformer(
        [('col' + str(i), DictVectorizer(), i) for i in range(2)])
    ct.fit(X)
    assert_equal(ct.get_feature_names(), ['col0__a', 'col0__b', 'col1__c'])

    # passthrough transformers not supported
    ct = ColumnTransformer([('trans', 'passthrough', [0, 1])])
    ct.fit(X)
    assert_raise_message(
        NotImplementedError, 'get_feature_names is not yet supported',
        ct.get_feature_names)

    ct = ColumnTransformer([('trans', DictVectorizer(), 0)])
    ct.fit(X)
    assert_raise_message(
        NotImplementedError, 'get_feature_names is not yet supported',
        ct.get_feature_names)

    # drop transformer
    ct = ColumnTransformer(
        [('col0', DictVectorizer(), 0), ('col1', 'drop', 1)])
    ct.fit(X)
    assert_equal(ct.get_feature_names(), ['col0__a', 'col0__b'])


def test_column_transformer_special_strings():

    # one 'drop' -> ignore
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T
    ct = ColumnTransformer(
        [('trans1', Trans(), [0]), ('trans2', 'drop', [1])])
    exp = np.array([[0.], [1.], [2.]])
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)

    # all 'drop' -> return shape 0 array
    ct = ColumnTransformer(
        [('trans1', 'drop', [0]), ('trans2', 'drop', [1])])
    assert_array_equal(ct.fit(X_array).transform(X_array).shape, (3, 0))
    assert_array_equal(ct.fit_transform(X_array).shape, (3, 0))

    # 'passthrough'
    X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T
    ct = ColumnTransformer(
        [('trans1', Trans(), [0]), ('trans2', 'passthrough', [1])])
    exp = X_array
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)

    # None itself / other string is not valid
    for val in [None, 'other']:
        ct = ColumnTransformer(
            [('trans1', Trans(), [0]), ('trans2', None, [1])])
        assert_raise_message(TypeError, "All estimators should implement",
                             ct.fit_transform, X_array)
        assert_raise_message(TypeError, "All estimators should implement",
                             ct.fit, X_array)


def test_column_transformer_remainder():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_second = np.array([2, 4, 6]).reshape(-1, 1)
    X_res_both = X_array

    # default passthrough
    ct = ColumnTransformer([('trans', Trans(), [0])])
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)

    # specify to drop remaining columns
    ct = ColumnTransformer([('trans1', Trans(), [0])],
                           remainder='drop')
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)

    # column order is not preserved (passed through added to end)
    ct = ColumnTransformer([('trans1', Trans(), [1])],
                           remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_both[:, ::-1])
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both[:, ::-1])

    # passthrough when all actual transformers are skipped
    ct = ColumnTransformer([('trans1', 'drop', [0])],
                           remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_second)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_second)

    # error on invalid arg
    ct = ColumnTransformer([('trans1', Trans(), [0])], remainder=1)
    assert_raise_message(
        ValueError,
        "remainder keyword needs to be one of \'drop\' or \'passthrough\'",
        ct.fit, X_array)
    assert_raise_message(
        ValueError,
        "remainder keyword needs to be one of \'drop\' or \'passthrough\'",
        ct.fit_transform, X_array)


@pytest.mark.parametrize("key", [[0], np.array([0]), slice(0, 1),
                                 np.array([True, False])])
def test_column_transformer_remainder_numpy(key):
    # test different ways that columns are specified with passthrough
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array

    ct = ColumnTransformer([('trans1', Trans(), key)],
                           remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)


@pytest.mark.parametrize(
    "key", [[0], slice(0, 1), np.array([True, False]), ['first'], 'pd-index',
            np.array(['first']), np.array(['first'], dtype=object),
            slice(None, 'first'), slice('first', 'first')])
def test_column_transformer_remainder_pandas(key):
    # test different ways that columns are specified with passthrough
    pd = pytest.importorskip('pandas')
    if isinstance(key, six.string_types) and key == 'pd-index':
        key = pd.Index(['first'])

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=['first', 'second'])
    X_res_both = X_array

    ct = ColumnTransformer([('trans1', Trans(), key)],
                           remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
