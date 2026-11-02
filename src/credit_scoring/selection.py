"""Turn per-fold scores into the evidence behind the choice of model."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

# Fixed before the scores are seen. Choosing the metric afterwards would be the
# same mistake as choosing the model on the test set.
SELECTION_METRIC = "pr_auc"


def summarise(per_fold: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Average each model across the folds and report how precise that average is.

    The standard error divides the spread by the square root of the fold count
    and answers what the average on its own cannot: how far it would move had
    the folds been drawn differently. Two models whose averages sit within one
    standard error of each other have not been told apart.
    """
    metrics = [column for column in per_fold.columns if column not in ("model", "fold")]
    grouped = per_fold.groupby("model", sort=False)

    table = grouped[metrics].mean()
    # Sample standard deviation: five folds are a sample of the splits that
    # could have been drawn, not the population being described.
    table[f"{metric}_std"] = grouped[metric].std(ddof=1)
    table[f"{metric}_se"] = table[f"{metric}_std"] / grouped[metric].count() ** 0.5

    return table.reset_index().sort_values(metric, ascending=False, ignore_index=True)


def pairwise(per_fold: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Compare every pair of models fold by fold.

    Both models score the same folds, so subtracting within a fold cancels out
    whatever part of the score is down to that fold being easy or hard. What is
    left is the difference between the models, measured more precisely than two
    separately computed averages allow.
    """
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
                # A count of folds won assumes nothing about how the scores are
                # distributed, so it still holds where the standard error of
                # five observations is open to argument.
                "folds_won_by_a": int((difference > 0).sum()),
            }
        )

    return pd.DataFrame(rows)
