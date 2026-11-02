"""Measure whether isotonic calibration improves the chosen model.

Usage:
    python scripts/run_calibration.py
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from credit_scoring.calibration import (
    calibrate_model,
    expected_calibration_error,
    plot_reliability,
    reliability_table,
)
from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compute_metrics
from credit_scoring.training import train_model


def main() -> None:
    """Fit both versions of the model and report whether calibration changes anything."""
    cfg = load_config()
    split = split_data(load_dataset(cfg), cfg)
    name = cfg.selected_model

    print(f"Calibrating {name} with {cfg.calibration.method} regression, cv={cfg.calibration.cv}")
    raw = train_model(name, split.X_train, split.y_train, cfg)
    calibrated = calibrate_model(split.X_train, split.y_train, cfg)

    raw_proba = raw.predict_proba(split.X_test)[:, 1]
    calibrated_proba = calibrated.predict_proba(split.X_test)[:, 1]

    before = reliability_table(split.y_test, raw_proba, cfg)
    after = reliability_table(split.y_test, calibrated_proba, cfg)

    comparison = pd.DataFrame(
        [
            {
                "model": "uncalibrated",
                **compute_metrics(split.y_test, raw_proba),
                "ece": expected_calibration_error(before),
            },
            {
                "model": cfg.calibration.method,
                **compute_metrics(split.y_test, calibrated_proba),
                "ece": expected_calibration_error(after),
            },
        ]
    )
    print(f"\n{comparison.round(4).to_string(index=False)}")

   
    rank_correlation = spearmanr(raw_proba, calibrated_proba).statistic
    print(f"\nrank correlation, raw vs calibrated: {rank_correlation:.6f}")

    print(f"\nuncalibrated\n{before.round(4).to_string(index=False)}")
    print(f"\n{cfg.calibration.method}\n{after.round(4).to_string(index=False)}")

    plot_reliability(before, after, cfg)

    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(directory / "calibration_comparison.csv", index=False)
    after.to_csv(directory / "calibration_reliability.csv", index=False)


if __name__ == "__main__":
    main()
