"""
xai_explanations.py

Generates plain-English XAI narratives and bar-chart plots for
the top-N merge pairs.  Used by run_xai_global.py.

Simplified from original:
  - Removed load_model() (unused - models are never loaded here)
  - Merged row_fixed and row_m2n2 into a single joined row (as run_xai_global
    already does the join before calling these functions)
  - Removed redundant branching in explain_pair_global
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fixed_results(task: str) -> pd.DataFrame:
    return pd.read_csv(f"./results/merges/{task}/merge_results_new_eccm.csv")


def load_m2n2_results(task: str) -> pd.DataFrame:
    return pd.read_csv(f"./results/merges/{task}/m2n2_results.csv")


# ── XAI narrative ─────────────────────────────────────────────────────────────

def explain_pair(row: pd.Series, task: str) -> str:
    """
    Generate a plain-English explanation for a single merged pair.

    Args:
        row:  a row from the joined fixed + m2n2 DataFrame
              (produced by run_xai_global joining on model_a, model_b)
        task: "fraud" or "churn"

    Returns:
        Multi-sentence explanation string
    """
    a, b    = row["model_a"], row["model_b"]
    psc     = row["psc"]
    fsc     = row["fsc"]
    rsc     = row["rsc"]
    eccm    = row["eccm_fixed"]
    base_i  = row["improvement"]
    opt_r   = row["opt_best_ratio"]
    opt_i   = row["opt_improvement"]
    delta   = row["opt_vs_fixed"]

    agreement = "often agree" if fsc > 0.9 else "agree on most cases" if fsc > 0.65 else "frequently disagree"
    ranking   = "very similar" if rsc > 0.9 else "moderately similar"

    lines = [
        f"For {task}, models {a} and {b} were selected with ECCM={eccm:.3f}.",
        f"PSC={psc:.3f}, FSC={fsc:.3f}, RSC={rsc:.3f}: "
        f"predictions {agreement} and feature rankings are {ranking}.",
        (f"Fixed-ratio merging improved AUC by {base_i:.6f}." if base_i >= 0
         else f"Fixed-ratio merging reduced AUC by {abs(base_i):.6f}."),
        f"CMA-ES found an optimal blend ratio of {opt_r:.3f} for {a}.",
        (f"This gained an additional {delta:.6f} AUC over the fixed grid."
         if delta > 0 else
         f"The fixed grid was already near-optimal (CMA-ES delta = {delta:.6f})."),
    ]
    return " ".join(lines)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pair(row: pd.Series, task: str, out_dir: str):
    """
    Save two bar charts for a pair:
      1. PSC / FSC / RSC sub-metric scores
      2. Best parent AUC vs fixed-merge AUC vs CMA-ES-merge AUC
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    a, b = row["model_a"], row["model_b"]
    stem = f"{task}_{a}_{b}"

    # Sub-metrics bar
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["PSC", "FSC", "RSC"],
           [row["psc"], row["fsc"], row["rsc"]],
           color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{task} - {a} + {b}")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{stem}_metrics.png", dpi=150)
    plt.close(fig)

    # AUC comparison bar
    vals   = [row["best_parent_auc"], row["fixed_best_auc"], row["opt_best_auc"]]
    labels = ["Best parent", "Fixed merge", "CMA-ES merge"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylim(min(vals) - 0.001, max(vals) + 0.001)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"{task} - {a} + {b} AUC")
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{stem}_auc.png", dpi=150)
    plt.close(fig)