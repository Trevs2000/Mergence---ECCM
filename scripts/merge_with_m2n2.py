"""
merge_with_m2n2.py

CMA-ES blend ratio optimisation for top-N ECCM-selected model pairs.

Why CMA-ES (not scalar optimisation)?
  CMA-ES is retained for alignment with the M2N2 (Model Merging via
  Natural Evolution) framework explored in this thesis. It demonstrates
  that an evolutionary search can find better blend ratios than a fixed
  grid, addressing RQ3.

  The 2D padding workaround (CMA-ES requires dim >= 2) is acknowledged
  in the thesis methodology: only x[0] is used as the blend ratio;
  x[1] is a dummy variable that satisfies the dimension requirement.

  The original DualEngineMerger (CMA-ES + scalar fallback) has been
  removed. A fallback that silently overrides the evolutionary engine
  weakens the thesis argument. CMA-ES is used exclusively so that if it
  converges to a suboptimal ratio on a specific pair, that is an honest
  empirical finding, not an error to be hidden.

Usage:
    python scripts/merge_with_m2n2.py
"""

import os
from datetime import datetime
from pathlib import Path

import cma
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.select_top_pairs import select_top_pairs


# ── CMA-ES optimiser ──────────────────────────────────────────────────────────
class CMAESMerger:
    """
    Optimise the blend ratio r in [0,1] via CMA-ES.

    Objective: maximise AUC(r * P_a + (1-r) * P_b, y_val)
    CMA-ES minimises, so we minimise negative AUC.
    """

    def __init__(self, sigma0: float = 0.3, max_iter: int = 50, popsize: int = 10):
        self.sigma0   = sigma0
        self.max_iter = max_iter
        self.popsize  = popsize

    def optimise(self, model_a, model_b, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Returns:
            best_ratio    — optimal blend weight for model_a  (float in [0,1])
            best_auc      — AUC at optimal ratio
            n_evaluations — total fitness evaluations used
        """
        pa = model_a.predict_proba(X_val)[:, 1]
        pb = model_b.predict_proba(X_val)[:, 1]

        def neg_auc(x):
            r = np.clip(x[0], 0.0, 1.0)
            return -roc_auc_score(y_val, r * pa + (1 - r) * pb)

        opts = cma.CMAOptions()
        opts["maxiter"] = self.max_iter
        opts["popsize"] = self.popsize
        opts["bounds"]  = [[0.0, 0.0], [1.0, 1.0]]  # bounds for [ratio, dummy]
        opts["verbose"] = -9
        opts["seed"]    = 42

        # x[0] = blend ratio, x[1] = dummy (satisfies CMA-ES dim >= 2 requirement)
        es      = cma.CMAEvolutionStrategy([0.5, 0.5], self.sigma0, opts)
        n_evals = 0
        while not es.stop():
            solutions  = es.ask()
            es.tell(solutions, [neg_auc(s) for s in solutions])
            n_evals   += len(solutions)

        r = es.result
        return {
            "best_ratio":    float(np.clip(r.xbest[0], 0.0, 1.0)),
            "best_auc":      float(-r.fbest),
            "n_evaluations": int(n_evals),
        }


# ── Full M2N2 pipeline ────────────────────────────────────────────────────────
class M2N2Pipeline:
    """
    Runs CMA-ES on the top-N ECCM-ranked pairs and records results.
    """

    def __init__(
        self,
        models_dir: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str,
        sigma0: float = 0.3,
        max_iter: int = 50,
        popsize: int = 10,
    ):
        self.models_dir = models_dir
        self.X_val      = X_val
        self.y_val      = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.merger     = CMAESMerger(sigma0=sigma0, max_iter=max_iter, popsize=popsize)
        self._cache: dict = {}

    def _load(self, model_id: str):
        if model_id not in self._cache:
            self._cache[model_id] = joblib.load(
                Path(self.models_dir) / f"{model_id}.pkl"
            )
        return self._cache[model_id]

    def run(self, top_pairs: pd.DataFrame, fixed_csv: str, output_filename: str = "m2n2_results.csv",) -> pd.DataFrame:
        """
        Args:
            top_pairs: [model_a, model_b, eccm] from select_top_pairs()
            fixed_csv: path to merge_results_new_eccm.csv (for baseline AUC)

        Returns:
            DataFrame saved to m2n2_results.csv
        """
        fixed = pd.read_csv(fixed_csv)
        rows  = []

        for i, pair in top_pairs.iterrows():
            mid_a, mid_b = pair["model_a"], pair["model_b"]
            ma = self._load(mid_a)
            mb = self._load(mid_b)

            opt = self.merger.optimise(ma, mb, self.X_val, self.y_val)

            mask       = (fixed["model_a"] == mid_a) & (fixed["model_b"] == mid_b)
            best_fixed = float(fixed.loc[mask, "auc_merged"].max())
            auc_a      = roc_auc_score(self.y_val, ma.predict_proba(self.X_val)[:, 1])
            auc_b      = roc_auc_score(self.y_val, mb.predict_proba(self.X_val)[:, 1])
            best_par   = max(auc_a, auc_b)

            rows.append({
                "model_a":           mid_a,
                "model_b":           mid_b,
                "eccm":              pair["eccm"],
                "auc_a":             round(auc_a,          10),
                "auc_b":             round(auc_b,          10),
                "best_parent_auc":   round(best_par,       10),
                "fixed_best_auc":    round(best_fixed,     10),
                "fixed_improvement": round(best_fixed - best_par, 10),
                "opt_best_ratio":    opt["best_ratio"],
                "opt_best_auc":      round(opt["best_auc"], 10),
                "opt_improvement":   round(opt["best_auc"] - best_par, 10),
                "opt_vs_fixed":      round(opt["best_auc"] - best_fixed, 10),
                "opt_n_evals":       opt["n_evaluations"],
                "timestamp":         datetime.now().isoformat(),
            })

            delta = rows[-1]["opt_vs_fixed"]
            print(
                f"  {i+1:3d}. {mid_a} + {mid_b}  |  "
                f"Fixed: {best_fixed:.6f}  |  "
                f"CMA-ES: {opt['best_auc']:.6f} (r={opt['best_ratio']:.4f})  |  "
                f"Δ={delta:+.6f}"
            )

        df = pd.DataFrame(rows)
        out_path = Path(self.output_dir) / output_filename
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} results → {out_path}")
        return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Fraud
    fraud_df = pd.read_csv("./data/fraud_preprocessed.csv")
    X_f = fraud_df.drop("Class", axis=1).values
    y_f = fraud_df["Class"].values
    _, X_val_f, _, y_val_f = train_test_split(
        X_f, y_f, test_size=0.2, random_state=0, stratify=y_f
    )
    top_f = select_top_pairs("./results/merges/fraud/merge_results_new_eccm.csv", 50)
    M2N2Pipeline("./models/fraud", X_val_f, y_val_f, "./results/merges/fraud").run(
        top_f, "./results/merges/fraud/merge_results_new_eccm.csv"
    )

    # Churn
    churn_val = pd.read_csv("./data/churn_val_with_churn_col.csv")
    X_val_c   = churn_val.drop("Churn", axis=1).values
    y_val_c   = churn_val["Churn"].values
    top_c = select_top_pairs("./results/merges/churn/merge_results_new_eccm.csv", 50)
    M2N2Pipeline("./models/churn", X_val_c, y_val_c, "./results/merges/churn").run(
        top_c, "./results/merges/churn/merge_results_new_eccm.csv"
    )