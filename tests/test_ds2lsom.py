import numpy as np
import pytest

from ds2l_som.ds2lsom import DS2LSOM


class TestFit:
    def test_returns_self(self, X_blobs):
        model = DS2LSOM(n_prototypes=20)
        assert model.fit(X_blobs) is model

    def test_sets_labels_attr(self, X_blobs):
        model = DS2LSOM(n_prototypes=20).fit(X_blobs)
        assert hasattr(model, "labels_")

    def test_method_kmeans(self, X_blobs):
        model = DS2LSOM(n_prototypes=20, method="kmeans").fit(X_blobs)
        assert hasattr(model, "labels_")


class TestPredict:
    def test_output_shape(self, fitted_model, X_blobs):
        labels = fitted_model.predict(X_blobs)
        assert labels.shape == (len(X_blobs),)

    def test_labels_are_integers(self, fitted_model, X_blobs):
        labels = fitted_model.predict(X_blobs)
        assert labels.dtype.kind == "i"

    def test_valid_label_range(self, fitted_model, X_blobs):
        labels = fitted_model.predict(X_blobs)
        assert set(labels).issubset(set(range(-1, labels.max() + 1)))


class TestFitPredict:
    def test_consistent_with_fit_then_predict(self, X_blobs):
        m1 = DS2LSOM(n_prototypes=20, random_state=42)
        labels_fp = m1.fit_predict(X_blobs)
        labels_p = m1.predict(X_blobs)
        np.testing.assert_array_equal(labels_fp, labels_p)


class TestParams:
    def test_invalid_method_raises(self, X_blobs):
        with pytest.raises((ValueError, KeyError)):
            DS2LSOM(method="invalid").fit(X_blobs)

    def test_high_threshold_no_crash(self, X_blobs):
        DS2LSOM(n_prototypes=20, threshold=50).fit(X_blobs)
