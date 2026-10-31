"""Build the estimator and the full pipeline for each model."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_scoring.config import Config, load_config
from credit_scoring.preprocessing import preprocessing_steps

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}


def build_estimator(
    name: str, config: Config | None = None, params: dict | None = None
) -> BaseEstimator:
    """Create the estimator for ``name``.

    ``params`` overrides the tuned values stored in the config, which is what
    each Optuna trial needs while it is still searching.
    """
    cfg = config or load_config()
    if name not in ESTIMATORS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(ESTIMATORS)}")

    model_config = cfg.models[name]
    chosen = params if params is not None else model_config.best_params
    return ESTIMATORS[name](
        random_state=cfg.random_state, **{**model_config.fixed_params, **chosen}
    )


def build_pipeline(name: str, config: Config | None = None, params: dict | None = None) -> Pipeline:
    """Return the preprocessing steps for ``name`` followed by its estimator.

    Keeping preprocessing and estimator in one object means cross-validation
    refits both on each fold, so the transformations never see the data the
    model is scored on.
    """
    cfg = config or load_config()
    steps = preprocessing_steps(cfg.models[name], cfg)
    return Pipeline([*steps, ("model", build_estimator(name, cfg, params))])
