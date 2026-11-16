"""Turn per-fold scores into the evidence behind the choice of model."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

SELECTION_METRIC = "pr_auc"


def summarise(per_fold: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Average each model across the folds and report how precise that average is."""
    metrics = [column for column in per_fold.columns if column not in ("model", "fold")]
    grouped = per_fold.groupby("model", sort=False)

    table = grouped[metrics].mean()

    table[f"{metric}_std"] = grouped[metric].std(ddof=1)
    table[f"{metric}_se"] = table[f"{metric}_std"] / grouped[metric].count() ** 0.5

    return table.reset_index().sort_values(metric, ascending=False, ignore_index=True)


def pairwise(per_fold: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Compare every pair of models fold by fold."""
    wide = per_fold.pivot(index="fold", columns="model", values=metric)
    rows = []

    for left, right in combinations(wide.columns, 2):
        difference = wide[left] - wide[right]
        rows.append(
            {
                "model_a": left,
                "model_b": right,
                "mean_difference": difference.mean(),
                "std_error": difference.std(ddof=1) / len(difference) ** 0.5,

                "folds_won_by_a": int((difference > 0).sum()),
            }
        )

    return pd.DataFrame(rows)
