"""
train_fraud_models.py - Train all fraud RF variants.

Model categories:
  v00–v23   Main variants (used for merging experiments)
  v100–v104 Benchmark RF  (same hyperparams, different seeds - used for evaluation only)
  Logistic  Baseline      (evaluation only, not saved as pkl)

The embedded X_train_sample_ attribute (200 stratified rows) is attached to
every saved model so the Streamlit app can compute FSC without a CSV upload.
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


# 24 hyperparameter variants for the main merge experiment
VARIANTS = [
    (50, 10, 5), (50, 10, 10), (50, 15, 5),  (50, 15, 10), (50, 20, 10),
    (100,10, 5), (100,10,10),  (100,15, 5),  (100,15,10),  (100,20,10),
    (150,10, 5), (150,10,20),  (150,15,10),  (150,15,20),  (150,20, 5),
    (200,10,10), (200,10,20),  (200,15, 5),  (200,15,20),  (200,20,10),
    (75, 12, 8), (125,12,15),  (175,18,12),  (225,20,15),
]


class FraudModelTrainer:

    def __init__(self, data_path: str, output_dir: str = "./models/fraud"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        df      = pd.read_csv(data_path)
        self.X  = df.drop("Class", axis=1).values
        self.y  = df["Class"].values
        self.metadata: list = []

    # ── Core training helper ──────────────────────────────────────────────────

    def _fit_and_save(
        self,
        variant_id: int,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
    ) -> dict:
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X, self.y, test_size=0.2,
            random_state=variant_id, stratify=self.y,
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=variant_id,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_tr, y_tr)

        # Embed 200 stratified training rows for FSC fallback (Problem 1 fix)
        _, X_sample, _, _ = train_test_split(
            X_tr, y_tr,
            test_size=min(200, len(X_tr)) / len(X_tr),
            stratify=y_tr, random_state=42,
        )
        model.X_train_sample_ = X_sample.astype(np.float32)

        auc  = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        path = f"{self.output_dir}/fraud_v{variant_id:02d}.pkl"
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
        """Train v00–v23 - these are the merge candidates."""
        print(f"\nTraining {len(VARIANTS)} main fraud variants...\n")
        for i, (n, d, s) in enumerate(VARIANTS):
            self.metadata.append(self._fit_and_save(i, n, d, s))
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        return self.metadata

    def train_benchmark_variants(
        self,
        n_estimators: int = 150,
        max_depth: int = 10,
        min_samples_split: int = 5,
        num_runs: int = 5,
    ) -> list:
        """
        Train v100–v104: same hyperparams, different seeds.
        Purpose: measure pre-merge stability and provide post-merge comparison.
        These models are NEVER used as merge candidates.
        """
        print(f"\nTraining {num_runs} benchmark variants (evaluation only)...\n")
        bench_meta = []
        for run in range(num_runs):
            bench_meta.append(self._fit_and_save(100 + run, n_estimators, max_depth, min_samples_split))
        return bench_meta

    def evaluate_logistic_baseline(self, variant_id: int = 999) -> dict:
        """
        Fit a logistic regression and report AUC.
        Evaluation only - not saved as pkl, not used in merging.
        """
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X, self.y, test_size=0.2,
            random_state=variant_id, stratify=self.y,
        )
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"  Logistic baseline AUC={auc:.4f}")
        return {"model_type": "logistic_regression", "auc": round(auc, 6)}

    def cross_validate(self, n_estimators=150, max_depth=10, min_samples_split=5, n_splits=5):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc = cross_val_score(model, self.X, self.y, cv=cv, scoring="roc_auc")
        print(f"  CV AUC: mean={auc.mean():.4f}  std={auc.std():.4f}")
        return auc


if __name__ == "__main__":
    trainer = FraudModelTrainer(
        data_path="./data/fraud_preprocessed.csv",
        output_dir="./models/fraud",
    )
    trainer.train_main_variants()
    trainer.train_benchmark_variants()
    trainer.evaluate_logistic_baseline()
    trainer.cross_validate(n_estimators=150, max_depth=10, min_samples_split=5)