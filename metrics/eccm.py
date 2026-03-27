"""
eccm.py — Evolutionary Compatibility & Co-evolution Metric (ECCM)

ECCM = w_psc × PSC + w_fsc × FSC + w_rsc × RSC + w_epc × EPC

Key design decisions:
  - Task-specific weights (fraud vs churn) derived from per-task EPC training
  - FSC fallback chain: real CSV → embedded sample → synthetic → imputed
  - Data-driven tier thresholds via isotonic regression on 276-pair results
  - EPC via contextual k-NN (not a frozen regressor)
"""

import numpy as np
from typing import Optional
from .psc import PSCCalculator
from .fsc import FSCCalculator
from .rsc import RSCCalculator
from .epc import EPCTrainer


# ── Task-specific weights ─────────────────────────────────────────────────────
# Derived from separate per-task RandomForest EPC models.
# Fraud: FSC dominates (0.526) — prediction agreement is the strongest
#        predictor of merge success when models vary widely in behaviour.
# Churn: more balanced — churn models are structurally very similar, so
#        PSC and RSC carry more signal than FSC alone.
TASK_WEIGHTS = {
    "fraud":   {"w_psc": 0.284, "w_fsc": 0.526, "w_rsc": 0.190, "w_epc": 0.0},
    "churn":   {"w_psc": 0.389, "w_fsc": 0.251, "w_rsc": 0.360, "w_epc": 0.0},
    "unknown": {"w_psc": 0.156, "w_fsc": 0.743, "w_rsc": 0.101, "w_epc": 0.0},
}

# ── Data-driven tier thresholds ───────────────────────────────────────────────
# Derived from isotonic regression on per-pair ECCM vs success-rate data.
# P(success) crosses ~0.55 at the low→medium boundary,
# P(success) crosses ~0.80 at the medium→high boundary.
TIER_THRESHOLDS = {
    "fraud":   {"low_to_med": 0.843, "med_to_high": 0.935},
    "churn":   {"low_to_med": 0.960, "med_to_high": 0.988},
    "unknown": {"low_to_med": 0.650, "med_to_high": 0.850},
}


def get_tier(eccm_score: float, task: str = "unknown") -> tuple[str, str, str]:
    """Return (label, hex_colour, emoji) using data-driven thresholds."""
    t = TIER_THRESHOLDS.get(task, TIER_THRESHOLDS["unknown"])
    if eccm_score >= t["med_to_high"]:
        return "High Compatibility",   "#28a745", "✅"
    elif eccm_score >= t["low_to_med"]:
        return "Medium Compatibility", "#ffc107", "⚠️"
    else:
        return "Low Compatibility",    "#dc3545", "❌"


def get_success_probability(eccm_score: float, task: str = "unknown") -> float:
    """Piecewise-linear estimate of P(merge success) from calibration curve."""
    t    = TIER_THRESHOLDS.get(task, TIER_THRESHOLDS["unknown"])
    low  = t["low_to_med"]
    high = t["med_to_high"]
    if eccm_score >= high:
        return float(np.clip(0.80 + 0.20 * (eccm_score - high) / max(1.0 - high, 1e-6), 0.80, 1.0))
    elif eccm_score >= low:
        return float(np.clip(0.40 + 0.40 * (eccm_score - low)  / max(high - low,  1e-6), 0.40, 0.80))
    else:
        return float(np.clip(0.40 * eccm_score / max(low, 1e-6), 0.0, 0.40))


# ── Synthetic validation data from tree thresholds ────────────────────────────
def synthetic_validation_from_rf(model, n_samples: int = 500) -> np.ndarray:
    """
    Generate a synthetic feature array from a RandomForest's split thresholds.

    Used as a fallback when no validation CSV is provided and the model
    has no embedded X_train_sample_.  FSC only measures agreement between
    two models (not ground-truth AUC), so synthetic data is sufficient.
    """
    n_features  = model.n_features_in_
    feat_min    = np.full(n_features,  np.inf)
    feat_max    = np.full(n_features, -np.inf)

    for tree in model.estimators_:
        t = tree.tree_
        for fi, th in zip(t.feature, t.threshold):
            if fi < 0:
                continue
            feat_min[fi] = min(feat_min[fi], th)
            feat_max[fi] = max(feat_max[fi], th)

    no_splits             = feat_min == np.inf
    feat_min[no_splits]   = 0.0
    feat_max[no_splits]   = 1.0
    margin                = (feat_max - feat_min) * 0.05
    feat_min             -= margin
    feat_max             += margin

    return np.random.default_rng(42).uniform(
        feat_min, feat_max, size=(n_samples, n_features)
    ).astype(np.float32)


# ── Main ECCM calculator ──────────────────────────────────────────────────────
class ECCMCalculator:

    def __init__(self, task: str = "unknown"):
        self.task     = task
        w             = TASK_WEIGHTS.get(task, TASK_WEIGHTS["unknown"])
        self.w_psc    = w["w_psc"]
        self.w_fsc    = w["w_fsc"]
        self.w_rsc    = w["w_rsc"]
        self.w_epc    = w["w_epc"]
        self.psc_calc = PSCCalculator(method="cosine")
        self.fsc_calc = FSCCalculator(strategy="correlation")
        self.rsc_calc = RSCCalculator()
        self.epc      = EPCTrainer()

    def compute(
        self,
        model_a,
        model_b,
        X: Optional[np.ndarray] = None,
        epc_pred: float = 0.0,
    ) -> dict:
        """
        Compute ECCM and all sub-metrics.

        Data resolution order for FSC:
          1. User-supplied X
          2. model_a.X_train_sample_  (embedded during training)
          3. Synthetic data from tree thresholds
          4. FSC imputed from k-NN history (last resort)

        Returns dict with: psc, fsc, rsc, epc, eccm, data_mode,
                           epc_reliability, epc_neighbours,
                           tier, tier_colour, tier_emoji, p_success, weights
        """
        # ── Resolve data ──────────────────────────────────────────────────────
        data_mode = "full"
        if X is None or len(X) == 0:
            X = getattr(model_a, "X_train_sample_", None)
            data_mode = "embedded" if X is not None else "synthetic"
            if X is None:
                try:
                    X = synthetic_validation_from_rf(model_a)
                except Exception:
                    X         = None
                    data_mode = "pscrsc_only"

        # ── Sub-metrics ───────────────────────────────────────────────────────
        psc = self.psc_calc.compute(model_a, model_b)
        rsc = self.rsc_calc.compute(model_a, model_b)
        fsc = (
            self.fsc_calc.compute(model_a, model_b, X)
            if X is not None
            else self._impute_fsc(psc, rsc)
        )

        # ── EPC (k-NN contextual) ─────────────────────────────────────────────
        epc_reliability, epc_neighbours = 0.5, []
        if epc_pred == 0.0:
            try:
                epc_pred, epc_reliability, epc_neighbours = (
                    self.epc.predict_with_context(psc, fsc, rsc)
                )
            except Exception:
                epc_pred = 0.0

        # ── Composite score ───────────────────────────────────────────────────
        eccm = float(np.clip(
            self.w_psc * psc + self.w_fsc * fsc +
            self.w_rsc * rsc + self.w_epc * epc_pred,
            0.0, 1.0,
        ))

        tier, tier_colour, tier_emoji = get_tier(eccm, self.task)

        return {
            "psc":              float(psc),
            "fsc":              float(fsc),
            "rsc":              float(rsc),
            "epc":              float(epc_pred),
            "eccm":             eccm,
            "data_mode":        data_mode,
            "epc_reliability":  epc_reliability,
            "epc_neighbours":   epc_neighbours,
            "tier":             tier,
            "tier_colour":      tier_colour,
            "tier_emoji":       tier_emoji,
            "p_success":        get_success_probability(eccm, self.task),
            "weights": {
                "w_psc": self.w_psc, "w_fsc": self.w_fsc,
                "w_rsc": self.w_rsc, "w_epc": self.w_epc,
            },
        }

    def _impute_fsc(self, psc: float, rsc: float) -> float:
        """Last-resort FSC imputation from k-NN history using only PSC + RSC."""
        h = self.epc._history
        if h is None or len(h) == 0:
            return 0.5
        dists   = np.linalg.norm(h[:, [0, 2]] - np.array([psc, rsc]), axis=1)
        k       = min(self.epc.k, len(dists))
        nn_idx  = np.argsort(dists)[:k]
        weights = 1.0 / (dists[nn_idx] + self.epc.epsilon)
        return float(np.average(h[nn_idx, 1], weights=weights))