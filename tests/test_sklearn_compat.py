from sklearn.utils.estimator_checks import parametrize_with_checks

from ds2l_som.ds2lsom import DS2LSOM


@parametrize_with_checks([DS2LSOM()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
