"""
select_top_pairs.py - Select top-N model pairs by ECCM score.
Used as input to merge_with_m2n2.py.
"""

import pandas as pd


def select_top_pairs(results_csv: str, top_n: int = 50) -> pd.DataFrame:
    """
    Return the top-N unique pairs ranked by ECCM score (descending).

    ECCM is constant across blend ratios for a given pair, so duplicate
    rows are dropped before ranking.

    Args:
        results_csv: path to merge_results_new_eccm.csv
        top_n:       number of pairs to select

    Returns:
        DataFrame with columns [model_a, model_b, eccm]
    """
    df   = pd.read_csv(results_csv)
    uniq = df.drop_duplicates(subset=["model_a", "model_b"], keep="first")
    top  = uniq.nlargest(top_n, "eccm")[["model_a", "model_b", "eccm"]].reset_index(drop=True)

    print(f"Selected {len(top)} / {len(uniq)} pairs  "
          f"(ECCM {top['eccm'].min():.4f} – {top['eccm'].max():.4f})")
    return top
