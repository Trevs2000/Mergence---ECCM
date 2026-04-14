"""
run_xai_global.py - CLI runner for global XAI reports.

Joins fixed-ratio results with CMA-ES results, picks the top-N pairs
by ECCM, prints a narrative explanation, and saves bar-chart PNGs.
"""

import pandas as pd
from scripts.xai_explanantions import (
    load_fixed_results,
    load_m2n2_results,
    explain_pair,
    plot_pair,
)


def run_for_task(task: str, top_n: int = 5):
    fixed  = load_fixed_results(task)
    m2n2   = load_m2n2_results(task)

    #Join on model_a + model_b; suffixes disambiguate shared column names
    joined = fixed.merge(
        m2n2, on=["model_a", "model_b"], suffixes=("_fixed", "_opt")
    )
    top = joined.sort_values("eccm_fixed", ascending=False).head(top_n)

    for _, row in top.iterrows():
        print("\n--- " + task + ": " + row["model_a"] + " + " + row["model_b"] + " ---")
        print(explain_pair(row, task))
        plot_pair(row, task, out_dir="./results/xai/" + task)


if __name__ == "__main__":
    run_for_task("fraud", top_n=5)
    run_for_task("churn", top_n=5)