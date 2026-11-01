"""Run the exploratory analysis on the training set and save its tables and figures.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.eda import (
    correlation_with_target,
    counter_value_counts,
    debt_ratio_by_income_missingness,
    default_rate_by_band,
    default_rate_by_group,
    descriptive_statistics,
    duplicate_report,
    implausible_value_report,
    missing_per_row,
    outlier_report,
    plot_boxplots,
    plot_correlation_matrix,
    plot_distributions,
    plot_target_balance,
    save_table,
    structure_table,
)


def main() -> None:
    """Describe the training set, saving every table and figure to the reports directory."""
    cfg = load_config()
    
    frame = split_data(load_dataset(cfg), cfg).training_frame()

    tables = {
        "structure": structure_table(frame),
        "descriptive_statistics": descriptive_statistics(frame),
        "missing_per_row": missing_per_row(frame),
        "outliers": outlier_report(frame, cfg),
        "implausible_values": implausible_value_report(frame, cfg),
        "delinquency_value_counts": counter_value_counts(frame, cfg),
        "default_rate_by_group": default_rate_by_group(frame, cfg),
        "debt_ratio_by_income_missingness": debt_ratio_by_income_missingness(frame),
        "utilisation_bands": default_rate_by_band(
            frame,
            "RevolvingUtilizationOfUnsecuredLines",
            [0, 1, 2, 5, 10, 100, float("inf")],
            cfg,
        ),
        "duplicates": duplicate_report(frame, cfg),
        "correlation_with_target": correlation_with_target(frame, cfg),
    }

    for name, table in tables.items():
        path = save_table(table, name, cfg)
        print(f"\n{name}\n{table.round(4).to_string(index=False)}")
        print(f"saved: {path}")

    features = [column for column in frame.columns if column != cfg.data.target]
    
    skewed = cfg.preprocessing.winsorize.columns

    plot_target_balance(frame, cfg)
    plot_distributions(frame, features, config=cfg)
    plot_distributions(frame, skewed, trim_quantile=0.99, config=cfg)
    plot_boxplots(frame, features, zoom_quantile=0.99, config=cfg)
    plot_correlation_matrix(frame, cfg)
    print(f"\nfigures saved to: {cfg.paths.absolute(cfg.paths.figures_dir)}")


if __name__ == "__main__":
    main()
