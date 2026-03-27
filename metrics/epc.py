"""
epc.py — Evolutionary Pressure Compatibility (EPC)

Uses k-Nearest Neighbours over merge history instead of a frozen RF regressor.

Why k-NN?
  - Transparent: the exact historical merges used as evidence can be shown to users
  - Self-updating: add a row to the history CSV and EPC immediately improves
  - Honest uncertainty: if the query pair is unlike any historical pair,
    the reliability score drops and the UI shows a warning
  - No retraining needed — ever

The RF model is still trained internally so that feature_importances_ can be
extracted for ECCM weight derivation (train_epc.py reads these weights).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


class EPCTrainer:

    def __init__(self, k: int = 5, epsilon: float = 1e-6):
        self.k        = k
        self.epsilon  = epsilon
        self._history: np.ndarray | None = None   # shape (n, 4): [PSC, FSC, RSC, improvement]
        self._history_df: pd.DataFrame | None = None
        self._rf_model: RandomForestRegressor | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, merge_history: pd.DataFrame, n_trees: int = 100) -> float:
        """
        Load merge history into the k-NN store and train the RF fallback.

        Args:
            merge_history: DataFrame with columns psc, fsc, rsc, improvement
            n_trees:       trees for the RF fallback

        Returns:
            RF train R² (for reporting only)
        """
        X = merge_history[["psc", "fsc", "rsc"]].values.astype(float)
        y = merge_history["improvement"].values.astype(float)

        self._history    = np.column_stack([X, y])
        self._history_df = merge_history.reset_index(drop=True)

        self._rf_model = RandomForestRegressor(
            n_estimators=n_trees, max_depth=8, random_state=42, n_jobs=-1
        )
        self._rf_model.fit(X, y)
        r2 = self._rf_model.score(X, y)
        print(f"  EPC: {len(merge_history)} merges loaded | RF Train R²={r2:.4f}")
        return r2

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_with_context(
        self, psc: float, fsc: float, rsc: float
    ) -> tuple[float, float, list[dict]]:
        """
        k-NN contextual prediction.

        Returns:
            epc_pred    — weighted average improvement of k neighbours
            reliability — 0..1; low = query is far from all history
            neighbours  — list of dicts (shown as evidence table in the UI)
        """
        if self._history is None or len(self._history) == 0:
            # RF fallback if history was never loaded
            if self._rf_model is not None:
                pred = float(self._rf_model.predict([[psc, fsc, rsc]])[0])
                return pred, 0.5, []
            return 0.0, 0.0, []

        query   = np.array([psc, fsc, rsc])
        X_hist  = self._history[:, :3]
        y_hist  = self._history[:, 3]
        dists   = np.linalg.norm(X_hist - query, axis=1)

        k       = min(self.k, len(dists))
        nn_idx  = np.argsort(dists)[:k]
        weights = 1.0 / (dists[nn_idx] + self.epsilon)
        epc_pred = float(np.average(y_hist[nn_idx], weights=weights))

        median_spread = float(np.median(np.linalg.norm(X_hist - X_hist.mean(0), axis=1)))
        reliability   = float(np.clip(
            1.0 / (1.0 + dists[nn_idx].mean() / (median_spread + self.epsilon)),
            0.0, 1.0,
        ))

        neighbours = []
        w_total = weights.sum()
        for rank, idx in enumerate(nn_idx):
            row: dict = {
                "rank":        rank + 1,
                "psc":         float(X_hist[idx, 0]),
                "fsc":         float(X_hist[idx, 1]),
                "rsc":         float(X_hist[idx, 2]),
                "improvement": float(y_hist[idx]),
                "distance":    float(dists[idx]),
                "weight":      float(weights[rank] / w_total),
            }
            if self._history_df is not None:
                for col in ("model_a", "model_b", "blend_ratio", "success"):
                    if col in self._history_df.columns:
                        row[col] = self._history_df.iloc[idx][col]
            neighbours.append(row)

        return epc_pred, reliability, neighbours

    def predict(self, psc: float, fsc: float, rsc: float) -> float:
        pred, _, _ = self.predict_with_context(psc, fsc, rsc)
        return pred

    # ── Feature importances (for ECCM weight derivation) ─────────────────────

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._rf_model.feature_importances_ if self._rf_model else None

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        obj = joblib.load(path)
        if isinstance(obj, EPCTrainer):
            self.__dict__.update(obj.__dict__)
        elif isinstance(obj, RandomForestRegressor):
            # Backward-compatible: old pkl was a bare RF
            self._rf_model = obj
            self._history  = None
            self._history_df = None