"""
Tests for the BlendedModel sklearn contract.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BlendedModel(BaseEstimator, ClassifierMixin):
    """Mirror of BlendedModel in streamlit_app.py."""

    def __init__(self, model_a=None, model_b=None, ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.ratio = ratio

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)[:, 1]
        pb = self.model_b.predict_proba(X)[:, 1]
        p1 = self.ratio * pa + (1 - self.ratio) * pb
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        return getattr(self.model_a, "classes_", np.array([0, 1]))

    @property
    def n_features_in_(self):
        return getattr(self.model_a, "n_features_in_", None)

    @property
    def feature_importances_(self):
        fa = self.model_a.feature_importances_
        fb = self.model_b.feature_importances_
        return self.ratio * fa + (1 - self.ratio) * fb

    @property
    def X_train_sample(self):
        return getattr(self.model_a, "X_train_sample", None)


def test_predict_proba_shape(model_a, model_b, val_X):
    m = BlendedModel(model_a, model_b, 0.6)
    assert m.predict_proba(val_X).shape == (len(val_X), 2)


def test_predict_proba_sums_to_one(model_a, model_b, val_X):
    m = BlendedModel(model_a, model_b, 0.5)
    np.testing.assert_allclose(m.predict_proba(val_X).sum(axis=1), 1.0, atol=1e-6)


def test_predict_outputs_are_binary(model_a, model_b, val_X):
    m = BlendedModel(model_a, model_b, 0.5)
    assert set(m.predict(val_X)).issubset({0, 1})


def test_feature_importances_non_negative(model_a, model_b):
    m = BlendedModel(model_a, model_b, 0.5)
    assert np.all(m.feature_importances_ >= 0)


def test_has_required_sklearn_attributes(model_a, model_b):
    m = BlendedModel(model_a, model_b, 0.5)
    assert hasattr(m, "classes_")
    assert hasattr(m, "n_features_in_")
    assert hasattr(m, "X_train_sample")


def test_ratio_zero_equals_model_b_output(model_a, model_b, val_X):
    m = BlendedModel(model_a, model_b, ratio=0.0)
    pb = model_b.predict_proba(val_X)[:, 1]
    np.testing.assert_allclose(m.predict_proba(val_X)[:, 1], pb, atol=1e-6)


def test_ratio_one_equals_model_a_output(model_a, model_b, val_X):
    m = BlendedModel(model_a, model_b, ratio=1.0)
    pa = model_a.predict_proba(val_X)[:, 1]
    np.testing.assert_allclose(m.predict_proba(val_X)[:, 1], pa, atol=1e-6)