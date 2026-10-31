"""Cross-validate the tuned models and save the comparison table.

Usage:
    python scripts/run_comparison.py
"""

from __future__ import annotations

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compare_models

TABLE_FILE = "cv_model_comparison.csv"


def main() -> None:
    """Score every tuned model on the shared folds and write the table to disk."""
    cfg = load_config()
    split = split_data(load_dataset(cfg), cfg)

    table = compare_models(split.X_train, split.y_train, cfg)
    print(table.round(4).to_string(index=False))

    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABLE_FILE
    table.to_csv(path, index=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
