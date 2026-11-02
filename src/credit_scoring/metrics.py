"""Scoring metrics and cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from credit_scoring.config import Config, load_config
from credit_scoring.models import build_pipeline


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    """Return the five metrics."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "ks": ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic,
        "brier": brier_score_loss(y_true, y_proba),
    }


def folds(config: Config | None = None) -> StratifiedKFold:
    """Return the fold definition shared by every cross-validation.

    The same folds are used for every model.
    """
    cfg = config or load_config()
    return StratifiedKFold(
        n_splits=cfg.cross_validation.n_splits,
        shuffle=cfg.cross_validation.shuffle,
        random_state=cfg.random_state,
    )


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, config: Config | None = None
) -> np.ndarray:
    """Return one score per fold for ``pipeline``."""
    cfg = config or load_config()
    return cross_val_score(
        pipeline, X, y, cv=folds(cfg), scoring=cfg.cross_validation.scoring, n_jobs=1
    )


def compare_models(X: pd.DataFrame, y: pd.Series, config: Config | None = None) -> pd.DataFrame:
    """Cross-validate every configured model and return one row per model per fold."""
    cfg = config or load_config()
    splitter = folds(cfg)
    rows = []

    for name, model_config in cfg.models.items():
        pipeline = build_pipeline(name, cfg)

        for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
            
            fitted = clone(pipeline).fit(X.iloc[train_index], y.iloc[train_index])
            proba = fitted.predict_proba(X.iloc[valid_index])[:, 1]
            rows.append(
                {
                    "model": model_config.display_name,
                    "fold": fold,
                    **compute_metrics(y.iloc[valid_index], proba),
                }
            )

    return pd.DataFrame(rows)
