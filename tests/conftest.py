"""
conftest.py: shared pytest fixtures for the ECCM test suite.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def make_model(seed=1):
    X, y = make_classification(n_samples=200, n_features=8, random_state=seed)
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=seed)
    clf.fit(X, y)
    clf.X_train_sample = X[:30]
    return clf, X, y


@pytest.fixture
def model_a():
    clf, _, _ = make_model(seed=1)
    return clf

@pytest.fixture
def model_b():
    clf, _, _ = make_model(seed=2)
    return clf

@pytest.fixture
def val_X():
    _, X, _ = make_model(seed=99)
    return X

@pytest.fixture
def val_y():
    _, _, y = make_model(seed=99)
    return y

@pytest.fixture
def merge_history_df():
    """Synthetic merge history DataFrame matching EPCTrainer.train() expected columns."""
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "psc":         rng.uniform(0.5, 1.0, n),
        "fsc":         rng.uniform(0.5, 1.0, n),
        "rsc":         rng.uniform(0.5, 1.0, n),
        "improvement": rng.uniform(-0.002, 0.010, n),
        "model_a":     [f"fraud_v{i:02d}" for i in range(n)],
        "model_b":     [f"fraud_v{i+1:02d}" for i in range(n)],
        "blend_ratio": rng.choice([0.3, 0.4, 0.5, 0.6, 0.7], n),
        "success":     rng.integers(0, 2, n),
    })