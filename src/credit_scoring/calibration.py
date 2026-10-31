"""Probability calibration for the model that goes to production."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from credit_scoring.config import Config, load_config
from credit_scoring.models import build_pipeline
from credit_scoring.plotting import INK_MUTED, figure, series_color


def calibrate_model(
    X: pd.DataFrame, y: pd.Series, config: Config | None = None
) -> CalibratedClassifierCV:
    """Fit the chosen model wrapped in cross-validated isotonic calibration.

    Each fold fits the model on four fifths of the data and the calibration map
    on the fifth it never saw, so the map learns to correct out-of-sample
    probabilities rather than the overconfident ones a model gives on data it
    was trained on.
    """
    cfg = config or load_config()
    calibrated = CalibratedClassifierCV(
        build_pipeline(cfg.calibration.model, cfg),
        method=cfg.calibration.method,
        cv=cfg.calibration.cv,
    )
    return calibrated.fit(X, y)


def reliability_table(
    y_true: pd.Series, y_proba: np.ndarray, config: Config | None = None
) -> pd.DataFrame:
    """Return mean predicted against observed default rate per probability bin.

    Quantile bins hold a comparable number of applicants each, which matters
    because predicted probabilities pile up near zero and equal-width bins would
    leave the upper ones nearly empty.
    """
    cfg = config or load_config()
    frame = pd.DataFrame({"predicted": np.asarray(y_proba), "default": np.asarray(y_true)})

    if cfg.calibration.bin_strategy == "quantile":
        frame["bin"] = pd.qcut(frame["predicted"], cfg.calibration.n_bins, duplicates="drop")
    else:
        frame["bin"] = pd.cut(frame["predicted"], cfg.calibration.n_bins)

    table = (
        frame.groupby("bin", observed=True)
        .agg(
            applicants=("default", "size"),
            predicted=("predicted", "mean"),
            observed=("default", "mean"),
        )
        .reset_index()
    )
    # A positive gap means the model promises fewer defaults than actually occur.
    return table.assign(gap=table["observed"] - table["predicted"])


def plot_reliability(
    before: pd.DataFrame, after: pd.DataFrame, config: Config | None = None
) -> None:
    """Plot predicted against observed default rate, before and after calibration."""
    cfg = config or load_config()
    limit = max(
        before[["predicted", "observed"]].max().max(), after[["predicted", "observed"]].max().max()
    )

    with figure(
        "calibration_curve",
        title="Reliability of predicted default probabilities",
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        config=cfg,
    ) as ax:
        ax.plot(
            [0, limit], [0, limit], linestyle="--", color=INK_MUTED, label="Perfect calibration"
        )
        ax.plot(
            before["predicted"],
            before["observed"],
            marker="o",
            color=series_color(0),
            label="Uncalibrated",
        )
        ax.plot(
            after["predicted"],
            after["observed"],
            marker="o",
            color=series_color(1),
            label=cfg.calibration.method.capitalize(),
        )
        ax.legend()
