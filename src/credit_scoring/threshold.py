"""Cost-based accept/reject cut-off and its sensitivity to the cost assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from credit_scoring.config import Config, load_config
from credit_scoring.plotting import INK, SURFACE, figure, sequential_cmap


def operating_point(
    y_true: pd.Series, y_proba: np.ndarray, cutoff: float, cost_ratio: float
) -> dict[str, float]:
    """Return what rejecting every applicant above ``cutoff`` costs and achieves.

    Approval rate and the bad rate among those approved are the two numbers a
    credit committee reads first: one is the business the policy writes, the
    other is the risk it keeps.
    """
    predicted = (np.asarray(y_proba) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    approved = predicted == 0

    return {
        "cutoff": cutoff,
        "approval_rate": approved.mean(),
        "bad_rate_approved": np.asarray(y_true)[approved].mean(),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        # Expressed in units of one false positive, so the two error types are
        # comparable on a single scale.
        "total_cost": fp + cost_ratio * fn,
    }


def sensitivity_table(
    y_true: pd.Series, y_proba: np.ndarray, config: Config | None = None
) -> pd.DataFrame:
    """Recompute the operating point across the configured range of cost ratios.

    The ratio of losing a defaulted loan to losing a good customer's margin is
    an assumption, not a measurement, so the policy is reported across a range
    rather than at a single value.
    """
    cfg = config or load_config()
    rows = [
        {"cost_ratio": k, **operating_point(y_true, y_proba, 1.0 / (k + 1.0), k)}
        for k in cfg.threshold.sensitivity_k_grid
    ]
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    y_true: pd.Series, y_proba: np.ndarray, cutoff: float, config: Config | None = None
) -> None:
    """Draw the confusion matrix produced by the chosen cut-off.

    Cells are shaded by their share of the actual class rather than by raw
    count, because the 26,000 correctly approved applicants would otherwise
    leave the three cells that matter looking identical.
    """
    cfg = config or load_config()
    matrix = confusion_matrix(y_true, (np.asarray(y_proba) >= cutoff).astype(int))
    shares = matrix / matrix.sum(axis=1, keepdims=True)

    with figure(
        "confusion_matrix",
        title=f"Confusion matrix at cut-off {cutoff:.4f}",
        xlabel="Predicted",
        ylabel="Actual",
        config=cfg,
    ) as ax:
        ax.imshow(shares, cmap=sequential_cmap(), vmin=0.0, vmax=1.0)
        ax.set_xticks(range(2), ["Approve", "Reject"])
        ax.set_yticks(range(2), ["Non-default", "Default"])
        ax.grid(False)

        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}\n{shares[row, column]:.1%}",
                    ha="center",
                    va="center",
                    color=SURFACE if shares[row, column] > 0.5 else INK,
                )
