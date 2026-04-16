"""
train_churn_models.py - Train all churn RF variants.

Same structure as train_fraud_models.py:
  v00–v23   Main variants  (merge candidates)
  v100–v104 Benchmark RF   (evaluation only)
  Logistic  Baseline       (evaluation only)
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


VARIANTS = [
    (50, 10, 5), (50, 10, 10), (50, 15, 5),  (50, 15, 10), (50, 20, 10),
    (100,10, 5), (100,10,10),  (100,15, 5),  (100,15,10),  (100,20,10),
    (150,10, 5), (150,10,20),  (150,15,10),  (150,15,20),  (150,20, 5),
    (200,10,10), (200,10,20),  (200,15, 5),  (200,15,20),  (200,20,10),
    (75, 12, 8), (125,12,15),  (175,18,12),  (225,20,15),
]


class ChurnModelTrainer:

    def __init__(self, train_path: str, val_path: str, output_dir: str = "./models/churn"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_df     = pd.read_csv(train_path)
        val_df       = pd.read_csv(val_path)
        self.X_train = train_df.drop("Churn", axis=1).values
        self.y_train = train_df["Churn"].values
        self.X_val   = val_df.drop("Churn", axis=1).values
        self.y_val   = val_df["Churn"].values
        self.metadata: list = []

    # ── Core training helper ──────────────────────────────────────────────────

    def _fit_and_save(
        self,
        variant_id: int,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
    ) -> dict:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=variant_id,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(self.X_train, self.y_train)

        # Embed 200 stratified training rows for FSC fallback
        _, X_sample, _, _ = train_test_split(
            self.X_train, self.y_train,
            test_size=min(200, len(self.X_train)) / len(self.X_train),
            stratify=self.y_train, random_state=42,
        )
        model.X_train_sample_ = X_sample.astype(np.float32)

        auc  = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
        path = f"{self.output_dir}/churn_v{variant_id:02d}.pkl"
        joblib.dump(model, path)

        meta = {
            "variant_id": variant_id,
            "hyperparams": {"n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split},
            "auc": round(auc, 6), "path": path,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  v{variant_id:02d}: AUC={auc:.4f} "
              f"(trees={n_estimators}, depth={max_depth}, split={min_samples_split})")
        return meta

    # ── Public methods ────────────────────────────────────────────────────────

    def train_main_variants(self) -> list:
        """Train v00–v23 - merge candidates."""
        print(f"\nTraining {len(VARIANTS)} main churn variants...\n")
        for i, (n, d, s) in enumerate(VARIANTS):
            self.metadata.append(self._fit_and_save(i, n, d, s))
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        return self.metadata

    def train_benchmark_variants(
        self,
        n_estimators: int = 50,
        max_depth: int = 15,
        min_samples_split: int = 10,
        num_runs: int = 5,
    ) -> list:
        """
        Train v100–v104: stability benchmarks.
        Evaluation only - never used as merge candidates.
        """
        print(f"\nTraining {num_runs} benchmark variants (evaluation only)...\n")
        bench_meta = []
        for run in range(num_runs):
            bench_meta.append(self._fit_and_save(100 + run, n_estimators, max_depth, min_samples_split))
        return bench_meta

    def evaluate_logistic_baseline(self) -> dict:
        """Fit logistic regression for evaluation. Not saved as pkl."""
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(self.X_train, self.y_train)
        auc = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
        print(f"  Logistic baseline AUC={auc:.4f}")
        return {"model_type": "logistic_regression", "auc": round(auc, 6)}

    def cross_validate(self, n_estimators=50, max_depth=15, min_samples_split=10, n_splits=5):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="roc_auc")
        print(f"  CV AUC: mean={auc.mean():.4f}  std={auc.std():.4f}")
        return auc


if __name__ == "__main__":
    trainer = ChurnModelTrainer(
        train_path="./data/churn_train_with_churn_col.csv",
        val_path="./data/churn_val_with_churn_col.csv",
        output_dir="./models/churn",
    )
    trainer.train_main_variants()
    trainer.train_benchmark_variants()
    trainer.evaluate_logistic_baseline()
    trainer.cross_validate(n_estimators=50, max_depth=15, min_samples_split=10)