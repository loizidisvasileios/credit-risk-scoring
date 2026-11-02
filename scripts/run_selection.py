"""Cross-validate the tuned models and justify the choice of the one to use in production.

Usage:
    python scripts/run_selection.py
"""

from __future__ import annotations

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compare_models
from credit_scoring.selection import SELECTION_METRIC, pairwise, summarise

SUBDIRECTORY = "selection"
PER_FOLD_FILE = "cv_per_fold.csv"
SUMMARY_FILE = "cv_summary.csv"
PAIRWISE_FILE = "cv_pairwise.csv"


def main() -> None:
    """Score every model on the shared folds."""
    cfg = load_config()
    split = split_data(load_dataset(cfg), cfg)

    print(
        f"Selecting on {SELECTION_METRIC} over {cfg.cross_validation.n_splits} folds "
        f"of {len(split.X_train):,} training rows, seed {cfg.random_state}"
    )

    per_fold = compare_models(split.X_train, split.y_train, cfg)
    summary = summarise(per_fold)
    pairs = pairwise(per_fold)

    folds = per_fold.pivot(index="fold", columns="model", values=SELECTION_METRIC)
    print(f"\n{SELECTION_METRIC} per fold\n{folds.round(4).to_string()}")
    print(f"\nsummary\n{summary.round(4).to_string(index=False)}")
    print(f"\npaired differences\n{pairs.round(4).to_string(index=False)}")

    directory = cfg.paths.absolute(cfg.paths.results_dir) / SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)

    tables = ((per_fold, PER_FOLD_FILE), (summary, SUMMARY_FILE), (pairs, PAIRWISE_FILE))
    print()

    for table, filename in tables:
        table.to_csv(directory / filename, index=False)
        print(f"saved: {directory / filename}")


if __name__ == "__main__":
    main()
