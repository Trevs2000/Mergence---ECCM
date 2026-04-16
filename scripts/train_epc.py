"""
train_epc.py - Train per-task EPC models and export ECCM weights.

Trains three EPCTrainer instances:
  - fraud-only   → epc_model_fraud.pkl
  - churn-only   → epc_model_churn.pkl
  - combined     → epc_model.pkl  (backward-compat fallback)

Feature importances from each RF are normalised to produce the
task-specific ECCM weights written to models/eccm_weights.json.

Usage:
    python scripts/train_epc.py
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from metrics.epc import EPCTrainer


def train_and_save(name: str, df: pd.DataFrame, path: str) -> dict:
    """Train one EPCTrainer, print metrics, save, and return ECCM weights."""
    print(f"\n{'='*55}")
    print(f"  {name}  ({len(df)} rows)")
    print(f"{'='*55}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    epc  = EPCTrainer(k=5)
    tr2  = epc.train(train_df, n_trees=100)
    te2  = epc._rf_model.score(
        test_df[["psc","fsc","rsc"]].values, test_df["improvement"].values
    )
    print(f"  Train R²={tr2:.4f}  |  Test R²={te2:.4f}")

    imp  = epc.feature_importances_
    total = imp.sum()
    w    = {"w_psc": float(imp[0]/total), "w_fsc": float(imp[1]/total),
            "w_rsc": float(imp[2]/total), "train_r2": tr2, "test_r2": te2}
    print(f"  Weights → PSC={w['w_psc']:.3f}  FSC={w['w_fsc']:.3f}  RSC={w['w_rsc']:.3f}")

    epc.save(path)
    print(f"  Saved → {path}")
    return w


if __name__ == "__main__":
    fraud = pd.read_csv("./results/merges/fraud/merge_results_new_eccm.csv")
    churn = pd.read_csv("./results/merges/churn/merge_results_new_eccm.csv")
    all_  = pd.concat([fraud, churn], ignore_index=True)

    weights = {
        "fraud":    train_and_save("Fraud",    fraud, "./models/epc_model_fraud.pkl"),
        "churn":    train_and_save("Churn",    churn, "./models/epc_model_churn.pkl"),
        "combined": train_and_save("Combined", all_,  "./models/epc_model.pkl"),
    }

    with open("./models/eccm_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print("\nWeights written → ./models/eccm_weights.json")