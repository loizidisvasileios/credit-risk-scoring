"""SHAP explanations for the deployed model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from shap import Explanation
from sklearn.pipeline import Pipeline

from credit_scoring.config import Config, load_config
from credit_scoring.plotting import SURFACE, apply_style, figure_path


def build_explainer(model: Pipeline) -> shap.TreeExplainer:
    """Build a TreeSHAP explainer for a fitted pipeline's booster.

    Construction walks every tree in the model, so a server scoring one
    applicant at a time builds this once and reuses it.
    """
    return shap.TreeExplainer(model.named_steps["model"])


def explain(
    model: Pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None
) -> Explanation:
    """Return exact TreeSHAP contributions for every row of ``X``.

    The rows are pushed through the pipeline's preprocessing first, because the
    explainer reads the booster directly and has to be handed the columns the
    booster was actually trained on.
    """
    explainer = explainer or build_explainer(model)
    return explainer(model[:-1].transform(X))


def _save(name: str, config: Config) -> None:
    """Save and close the figure that SHAP has just drawn into."""
    path = figure_path(name, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close()


def plot_importance(explanation: Explanation, config: Config | None = None) -> None:
    """Rank features by their mean absolute contribution across all applicants."""
    cfg = config or load_config()
    apply_style(cfg)

    shap.plots.bar(explanation, max_display=cfg.explainability.max_display, show=False)
    plt.title("XGBoost SHAP feature importance")
    _save("shap_importance", cfg)


def plot_beeswarm(explanation: Explanation, config: Config | None = None) -> None:
    """Show every applicant as a point, so direction and spread stay visible.

    The bar chart collapses each feature to one number; this keeps the sign of
    the effect and shows where a feature matters for a few applicants rather
    than slightly for all of them.
    """
    cfg = config or load_config()
    apply_style(cfg)

    shap.plots.beeswarm(explanation, max_display=cfg.explainability.max_display, show=False)
    plt.title("XGBoost SHAP summary")
    _save("shap_beeswarm", cfg)


def plot_waterfall(
    explanation: Explanation, index: int, label: str, config: Config | None = None
) -> None:
    """Break one applicant's prediction down from the average to their own score."""
    cfg = config or load_config()
    apply_style(cfg)

    shap.plots.waterfall(explanation[index], max_display=cfg.explainability.max_display, show=False)
    _save(f"shap_waterfall_{label}", cfg)


def reason_codes(explanation: Explanation, index: int, top_n: int = 3) -> pd.DataFrame:
    """Return the features that pushed one applicant's risk up the most.

    Only positive contributions are ranked, because a declined applicant has to
    be told what worked against them, not what was in their favour.
    """
    frame = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "value": explanation.data[index],
            "contribution": explanation.values[index],
        }
    )
    return frame.nlargest(top_n, "contribution").reset_index(drop=True)
