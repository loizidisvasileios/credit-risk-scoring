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
    """Return the five metrics used to judge a set of predicted probabilities.

    PR-AUC leads because the positive class is rare and is the one that matters;
    ROC-AUC and Gini are the same information in the form credit risk teams
    expect; KS summarises separation at its best point; Brier is the only one of
    the five that checks whether the probabilities are numerically right rather
    than merely well ordered.
    """
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
    """Return the fold definition shared by every cross-validation in the project.

    Tuning and model comparison have to score candidates on identical folds for
    their numbers to be comparable, so the splitter is defined in one place.
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
    """Return one score per fold for ``pipeline``.

    The whole pipeline is passed in, so preprocessing is refitted inside every
    fold and never sees the part it is scored on. Per-fold scores are returned
    rather than their mean so the caller can also report the spread.
    """
    cfg = config or load_config()
    return cross_val_score(
        pipeline, X, y, cv=folds(cfg), scoring=cfg.cross_validation.scoring, n_jobs=1
    )


def compare_models(X: pd.DataFrame, y: pd.Series, config: Config | None = None) -> pd.DataFrame:
    """Cross-validate every configured model and return all metrics per model.

    The fold loop is written out instead of delegating to ``cross_val_score``
    because that helper discards the predicted probabilities and returns only
    the single score it was asked for. Keeping the probabilities lets every
    metric be computed from the same predictions.
    """
    cfg = config or load_config()
    splitter = folds(cfg)
    rows = []

    for name, model_config in cfg.models.items():
        pipeline = build_pipeline(name, cfg)
        per_fold = []

        for train_index, valid_index in splitter.split(X, y):
            # A fresh clone per fold keeps preprocessing learned on the training
            # part only, with nothing carried over from the previous fold.
            fitted = clone(pipeline).fit(X.iloc[train_index], y.iloc[train_index])
            proba = fitted.predict_proba(X.iloc[valid_index])[:, 1]
            per_fold.append(compute_metrics(y.iloc[valid_index], proba))

        scores = pd.DataFrame(per_fold)
        rows.append(
            {
                "model": model_config.display_name,
                **scores.mean().to_dict(),
                # Population standard deviation, matching the spread recorded
                # for each tuning trial.
                "pr_auc_std": scores["pr_auc"].std(ddof=0),
            }
        )

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)
