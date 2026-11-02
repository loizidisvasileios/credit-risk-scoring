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
    """Build a TreeSHAP explainer."""
    return shap.TreeExplainer(model.named_steps["model"])


def explain(
    model: Pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None
) -> Explanation:
    """Return exact TreeSHAP contributions for every row of ``X``."""
    explainer = explainer or build_explainer(model)
    return explainer(model[:-1].transform(X))


def _save(name: str, config: Config) -> None:
    """Save and close the figure that SHAP has drawn."""
    path = figure_path(name, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close()


def plot_importance(explanation: Explanation, config: Config | None = None) -> None:
    """Rank features by their mean absolute contribution across all applicants."""
    cfg = config or load_config()
    apply_style(cfg)

    shap.plots.bar(explanation, max_display=cfg.explainability.max_display, show=False)
    
    plt.title(f"{cfg.models[cfg.selected_model].display_name} SHAP feature importance")
    _save("shap_importance", cfg)


def plot_beeswarm(explanation: Explanation, config: Config | None = None) -> None:
    """Show every applicant as a point, so direction and spread stay visible."""
    cfg = config or load_config()
    apply_style(cfg)

    shap.plots.beeswarm(explanation, max_display=cfg.explainability.max_display, show=False)
    plt.title(f"{cfg.models[cfg.selected_model].display_name} SHAP summary")
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
    """Return the features that pushed one applicant's risk up the most."""
    frame = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "value": explanation.data[index],
            "contribution": explanation.values[index],
        }
    )
    return frame.nlargest(top_n, "contribution").reset_index(drop=True)
