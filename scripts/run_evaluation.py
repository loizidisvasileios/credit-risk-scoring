"""Train the tuned models on the full training set and score them on the test set.

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import pandas as pd

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compute_metrics
from credit_scoring.training import train_model

TABLE_FILE = "test_model_comparison.csv"


def main() -> None:
    """Fit every configured model once and report its metrics on the test set."""
    cfg = load_config()
    split = split_data(load_dataset(cfg), cfg)
    rows = []

    for name, model_config in cfg.models.items():
        model = train_model(name, split.X_train, split.y_train, cfg)
        proba = model.predict_proba(split.X_test)[:, 1]
        rows.append({"model": model_config.display_name, **compute_metrics(split.y_test, proba)})


    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABLE_FILE
    table.to_csv(path, index=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
