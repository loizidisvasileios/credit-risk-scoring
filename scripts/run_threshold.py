"""Apply the cost-based cut-off to the test set and report its sensitivity.

Usage:
    python scripts/run_threshold.py
"""

from __future__ import annotations

import pandas as pd

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.threshold import operating_point, plot_confusion_matrix, sensitivity_table
from credit_scoring.training import load_model

TABLE_FILE = "threshold_sensitivity.csv"


def main() -> None:
    """Report the chosen operating point, then the same policy across cost ratios."""
    cfg = load_config()
    model, _ = load_model(cfg)
    split = split_data(load_dataset(cfg), cfg)

    proba = model.predict_proba(split.X_test)[:, 1]
    cutoff = cfg.threshold.optimal_cutoff
    k = cfg.threshold.cost_ratio

    print(f"LGD {cfg.threshold.loss_given_default}, margin {cfg.threshold.net_margin}")
    print(f"k = LGD / margin = {k:.3f}")
    print(f"cut-off = 1 / (k + 1) = {cutoff:.4f}\n")

    chosen = operating_point(split.y_test, proba, cutoff, k)
    for metric, value in chosen.items():
        print(f"  {metric:<19} {value:,.4f}")

    plot_confusion_matrix(split.y_test, proba, cutoff, cfg)

    table = sensitivity_table(split.y_test, proba, cfg)
    print(f"\n{table.round(4).to_string(index=False)}")

    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(directory / TABLE_FILE, index=False)
    print(f"\nsaved: {directory / TABLE_FILE}")


if __name__ == "__main__":
    main()
