"""
Tests for scripts/select_top_pairs.py and scripts/merge_with_m2n2.py.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from scripts.select_top_pairs import select_top_pairs
from scripts.merge_with_m2n2 import CMAESMerger


def make_csv(rows, tmp_path):
    p = tmp_path / "merge.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def make_rf(seed):
    X, y = make_classification(n_samples=150, n_features=8, random_state=seed)
    clf = RandomForestClassifier(n_estimators=10, random_state=seed).fit(X, y)
    clf.X_train_sample = X[:20]
    return clf, X, y


def test_select_top_pairs_deduplicates_ratios(tmp_path):
    rows = [
        {"model_a": "a", "model_b": "b", "eccm": 0.9, "blend_ratio": 0.3},
        {"model_a": "a", "model_b": "b", "eccm": 0.9, "blend_ratio": 0.5},
        {"model_a": "a", "model_b": "c", "eccm": 0.8, "blend_ratio": 0.4},
    ]
    out = select_top_pairs(make_csv(rows, tmp_path), top_n=10)
    assert len(out) == 2


def test_select_top_pairs_sorted_descending(tmp_path):
    rows = [
        {"model_a": "a", "model_b": "b", "eccm": 0.7},
        {"model_a": "c", "model_b": "d", "eccm": 0.95},
        {"model_a": "e", "model_b": "f", "eccm": 0.8},
    ]
    out = select_top_pairs(make_csv(rows, tmp_path), top_n=3)
    assert list(out["eccm"]) == sorted(out["eccm"], reverse=True)


def test_cmaes_best_ratio_is_valid():
    m1, X, y = make_rf(10)
    m2, _, _ = make_rf(11)
    merger = CMAESMerger(sigma0=0.3, max_iter=5, popsize=6)
    result = merger.optimise(m1, m2, X, y)
    assert 0.0 <= result["best_ratio"] <= 1.0


def test_cmaes_best_auc_is_valid():
    m1, X, y = make_rf(12)
    m2, _, _ = make_rf(13)
    merger = CMAESMerger(sigma0=0.3, max_iter=5, popsize=6)
    result = merger.optimise(m1, m2, X, y)
    assert 0.0 <= result["best_auc"] <= 1.0


def test_cmaes_returns_required_keys():
    m1, X, y = make_rf(14)
    m2, _, _ = make_rf(15)
    merger = CMAESMerger(sigma0=0.3, max_iter=5, popsize=6)
    result = merger.optimise(m1, m2, X, y)
    for key in ("best_ratio", "best_auc", "n_evaluations"):
        assert key in result