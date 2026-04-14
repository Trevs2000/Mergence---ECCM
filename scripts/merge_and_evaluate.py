"""
merge_and_evaluate.py

Fixed-ratio merge experiment for fraud and churn tasks.

Design decisions:
  - Only v00–v23 models are used as merge candidates.
  - Benchmark models (v100–v104) are evaluated separately via
    evaluate_baselines() — they are never merge candidates.
  - No BlendedModel wrapper class is needed here: we only need
    the blended probability array to compute AUC, so a one-liner
    blend_predict() replaces SimpleMerger entirely.
  - ECCM sub-metrics are computed ONCE per pair (they don't change
    across blend ratios), saving 4× redundant computation.
  - All global execution code is inside `if __name__ == "__main__"`
    so importing this module does not trigger any computation.

Usage:
    python scripts/merge_and_evaluate.py
"""

import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from metrics.eccm import ECCMCalculator

# ── Constants ─────────────────────────────────────────────────────────────────
BLEND_RATIOS        = [0.3, 0.4, 0.5, 0.6, 0.7]
MAIN_VARIANT_RANGE  = range(0, 24)    # v00–v23 are merge candidates
BENCH_VARIANT_RANGE = range(100, 105) # v100–v104 are evaluation-only


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_models_by_range(models_dir: str, variant_range) -> dict:
    """
    Load model pkl files whose numeric suffix falls within variant_range.

    Example:
        load_models_by_range('./models/fraud', range(0, 24))
        → {'fraud_v00': <RF>, 'fraud_v01': <RF>, ...}
    """
    models = {}
    for pkl in sorted(Path(models_dir).glob("*.pkl")):
        parts = pkl.stem.rsplit("_v", 1)
        if len(parts) != 2:
            continue
        try:
            vid = int(parts[1])
        except ValueError:
            continue
        if vid in variant_range:
            models[pkl.stem] = joblib.load(pkl)
    return models


# ── Core pipeline ─────────────────────────────────────────────────────────────
class MergePipeline:
    """
    Runs the fixed-ratio merge experiment for a single task.

    For every C(24,2)=276 unordered pair:
      1. Compute ECCM sub-metrics once per pair (PSC, FSC, RSC, ECCM)
      2. For each of 5 blend ratios: record merged AUC and improvement
      3. Save all 1380 rows to CSV
    """

    def __init__(
        self,
        models_dir: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: str = "unknown",
        output_dir: str = "./results/merges",
    ):
        self.models_dir = models_dir
        self.X_val      = X_val
        self.y_val      = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.eccm_calc  = ECCMCalculator(task=task)

    def run(self, num_pairs: int = 276) -> pd.DataFrame:
        """
        Run all merge experiments.

        Args:
            num_pairs: cap on number of pairs tested (max 276)

        Returns:
            DataFrame: one row per (pair × ratio), up to 1380 rows
        """
        models    = load_models_by_range(self.models_dir, MAIN_VARIANT_RANGE)
        pairs     = list(combinations(sorted(models.keys()), 2))[:num_pairs]

        print(f"\n{len(pairs)} pairs × {len(BLEND_RATIOS)} ratios = "
              f"{len(pairs) * len(BLEND_RATIOS)} experiments\n")

        rows = []
        for i, (mid_a, mid_b) in enumerate(pairs, 1):
            ma, mb = models[mid_a], models[mid_b]

            # ECCM metrics are the same for all blend ratios — compute once
            eccm = self.eccm_calc.compute(ma, mb, X=self.X_val)

            pa      = ma.predict_proba(self.X_val)[:, 1]
            pb      = mb.predict_proba(self.X_val)[:, 1]
            auc_a   = roc_auc_score(self.y_val, pa)
            auc_b   = roc_auc_score(self.y_val, pb)
            best    = max(auc_a, auc_b)

            for ratio in BLEND_RATIOS:
                auc_m = roc_auc_score(self.y_val, ratio * pa + (1 - ratio) * pb)
                impr  = auc_m - best
                rows.append({
                    "model_a":     mid_a,
                    "model_b":     mid_b,
                    "auc_a":       round(auc_a,  10),
                    "auc_b":       round(auc_b,  10),
                    "auc_merged":  round(auc_m,  10),
                    "improvement": round(impr,   10),
                    "success":     int(impr > 0),
                    "psc":         eccm["psc"],
                    "fsc":         eccm["fsc"],
                    "rsc":         eccm["rsc"],
                    "eccm":        eccm["eccm"],
                    "blend_ratio": ratio,
                    "timestamp":   datetime.now().isoformat(),
                })

            print(f"  {i:3d}. {mid_a} + {mid_b}")

        df       = pd.DataFrame(rows)
        csv_path = f"{self.output_dir}/merge_results_new_eccm.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows → {csv_path}")
        return df


# ── Baseline evaluation (evaluation-only, never merged) ───────────────────────
def evaluate_baselines(
    models_dir: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Evaluate benchmark RF models (v100–v104) on the validation set.

    Purpose: provide a pre/post-merge comparison baseline showing whether
    the merged model outperforms a stable single-seed RF variant.
    These models are NEVER used as merge candidates.

    Returns:
        DataFrame saved to baseline_results.csv
    """
    bench_models = load_models_by_range(models_dir, BENCH_VARIANT_RANGE)
    if not bench_models:
        print("  No benchmark models found (expected v100–v104).")
        return pd.DataFrame(columns=["model_id", "model_type", "auc", "task"])

    rows = []
    for mid, model in sorted(bench_models.items()):
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"model_id": mid, "model_type": "benchmark_rf",
                     "auc": round(auc, 10), "task": task})
        print(f"  Benchmark {mid}: AUC={auc:.6f}")

    df       = pd.DataFrame(rows)
    csv_path = f"{output_dir}/baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} baseline rows → {csv_path}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Fraud ──────────────────────────────────────────────────────────────────
    fraud_df = pd.read_csv("./data/fraud_preprocessed.csv")
    X_f = fraud_df.drop("Class", axis=1).values
    y_f = fraud_df["Class"].values
    _, X_val_f, _, y_val_f = train_test_split(
        X_f, y_f, test_size=0.2, random_state=0, stratify=y_f
    )

    MergePipeline(
        models_dir="./models/fraud",
        X_val=X_val_f,
        y_val=y_val_f,
        task="fraud",
        output_dir="./results/merges/fraud",
    ).run(num_pairs=276)

    evaluate_baselines(
        models_dir="./models/fraud",
        X_val=X_val_f,
        y_val=y_val_f,
        task="fraud",
        output_dir="./results/merges/fraud",
    )

    # ── Churn ──────────────────────────────────────────────────────────────────
    churn_val = pd.read_csv("./data/churn_val_with_churn_col.csv")
    X_val_c   = churn_val.drop("Churn", axis=1).values
    y_val_c   = churn_val["Churn"].values

    MergePipeline(
        models_dir="./models/churn",
        X_val=X_val_c,
        y_val=y_val_c,
        task="churn",
        output_dir="./results/merges/churn",
    ).run(num_pairs=276)

    evaluate_baselines(
        models_dir="./models/churn",
        X_val=X_val_c,
        y_val=y_val_c,
        task="churn",
        output_dir="./results/merges/churn",
    )