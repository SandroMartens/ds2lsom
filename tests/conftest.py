import pytest
from sklearn.datasets import make_blobs

from ds2l_som.ds2lsom import DS2LSOM


@pytest.fixture
def X_blobs():
    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    return X


@pytest.fixture
def fitted_model(X_blobs):
    return DS2LSOM(n_prototypes=20).fit(X_blobs)
