"""
Tests for metrics/psc.py, fsc.py, rsc.py, eccm.py.
"""
from metrics.psc import PSCCalculator
from metrics.fsc import FSCCalculator
from metrics.rsc import RSCCalculator
from metrics.eccm import ECCMCalculator


def test_psc_returns_value_between_0_and_1(model_a, model_b):
    score = PSCCalculator(method="cosine").compute(model_a, model_b)
    assert 0.0 <= float(score) <= 1.0


def test_fsc_returns_value_between_0_and_1(model_a, model_b, val_X):
    score = FSCCalculator(strategy="correlation").compute(model_a, model_b, val_X)
    assert 0.0 <= float(score) <= 1.0


def test_fsc_works_with_embedded_sample(model_a, model_b):
    # model_a.X_train_sample is set in conftest — FSC should not raise
    score = FSCCalculator(strategy="correlation").compute(
        model_a, model_b, model_a.X_train_sample
    )
    assert 0.0 <= float(score) <= 1.0


def test_rsc_returns_value_between_0_and_1(model_a, model_b):
    score = RSCCalculator().compute(model_a, model_b)
    assert 0.0 <= float(score) <= 1.0


def test_eccm_compute_returns_expected_keys(model_a, model_b, val_X):
    calc = ECCMCalculator(task="fraud")
    out = calc.compute(model_a, model_b, X=val_X)
    for key in ("psc", "fsc", "rsc", "eccm"):
        assert key in out


def test_eccm_score_between_0_and_1(model_a, model_b, val_X):
    calc = ECCMCalculator(task="fraud")
    out = calc.compute(model_a, model_b, X=val_X)
    assert 0.0 <= float(out["eccm"]) <= 1.0


def test_eccm_works_without_validation_data(model_a, model_b):
    calc = ECCMCalculator(task="fraud")
    out = calc.compute(model_a, model_b, X=None)
    assert "eccm" in out


def test_eccm_unknown_task_does_not_crash(model_a, model_b, val_X):
    calc = ECCMCalculator(task="unknown")
    out = calc.compute(model_a, model_b, X=val_X)
    assert "eccm" in out