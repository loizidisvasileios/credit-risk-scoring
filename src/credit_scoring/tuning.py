"""Hyperparameter search with Optuna."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.trial import Trial

from credit_scoring.config import Config, load_config
from credit_scoring.metrics import cross_validate_model
from credit_scoring.models import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: Trial, search_space: dict[str, dict[str, Any]], imbalance: float) -> dict:
    """Turn the search space declared in the config into one trial's parameters.

    Reading the space from configuration keeps a single generic objective for
    all four models instead of one hand-written function per model.
    """
    params: dict[str, Any] = {}

    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
            continue

        # The positive-class weight is bounded relative to the observed class
        # ratio, so the range adapts if the data changes.
        high = spec.get("high", spec.get("high_ratio_to_imbalance", 1.0) * imbalance)
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], int(high), step=spec.get("step", 1))
        else:
            params[name] = trial.suggest_float(name, spec["low"], high, log=spec.get("log", False))

    return params


def run_study(
    name: str, X: pd.DataFrame, y: pd.Series, config: Config | None = None
) -> optuna.Study:
    """Search the configured space for ``name`` and return the finished study.

    Each trial is scored by the mean of a 5-fold cross-validation, and the
    individual fold scores are stored on the trial so the spread is available
    afterwards without a second run.
    """
    cfg = config or load_config()
    imbalance = float((y == 0).sum() / (y == 1).sum())

    def objective(trial: Trial) -> float:
        params = suggest_params(trial, cfg.models[name].search_space, imbalance)
        scores = cross_validate_model(build_pipeline(name, cfg, params), X, y, cfg)
        trial.set_user_attr("fold_scores", [float(score) for score in scores])
        trial.set_user_attr("cv_std", float(scores.std()))
        return float(scores.mean())

    study = optuna.create_study(
        study_name=name,
        direction=cfg.tuning.direction,
        # A fixed seed makes the whole search reproducible. TPE samples the
        # first ten trials at random before it starts modelling the space.
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state, n_startup_trials=10),
    )
    study.optimize(objective, n_trials=cfg.tuning.n_trials)
    return study


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """Return every trial with its score and parameters, best first."""
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "trial": trial.number,
                "cv_mean": round(trial.value, 6),
                "cv_std": round(trial.user_attrs["cv_std"], 6),
                "seconds": round(
                    (trial.datetime_complete - trial.datetime_start).total_seconds(), 1
                ),
                **trial.params,
            }
        )
    return pd.DataFrame(rows).sort_values("cv_mean", ascending=False, ignore_index=True)


def save_trials(study: optuna.Study, config: Config | None = None) -> Path:
    """Write the full trial history to the results directory and return the path.

    Keeping every trial, not only the winner, is what makes the reported
    hyperparameters verifiable by someone reading the repository.
    """
    cfg = config or load_config()
    directory = cfg.paths.absolute(cfg.paths.results_dir) / "tuning"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{study.study_name}_trials.csv"
    trials_table(study).to_csv(path, index=False)
    return path


def _as_yaml(value: object) -> str:
    """Render a parameter so PyYAML reads it back as the same type."""
    if value is None:
        return "null"
    if isinstance(value, bool | int | str):
        return str(value).lower() if isinstance(value, bool) else str(value)
    # repr keeps full float precision, but an exponent without a decimal point
    # would be read back as a string.
    text = repr(float(value))
    mantissa, _, exponent = text.partition("e")
    return text if not exponent or "." in mantissa else f"{mantissa}.0e{exponent}"


def write_best_params(
    name: str, params: dict, cv_mean: float, config: Config | None = None
) -> None:
    """Replace a model's best_params block in config.yaml with a tuned set.

    The configuration stays the single source of truth for the pipeline, and the
    inserted comment records which run produced the values so they can be traced
    back to the saved trial history.
    """
    cfg = config or load_config()
    path = cfg.paths.absolute(Path("config/config.yaml"))
    lines = path.read_text(encoding="utf-8").splitlines()

    start = next(i for i, line in enumerate(lines) if line.rstrip() == f"  {name}:")
    block = next(i for i in range(start, len(lines)) if lines[i].strip() == "best_params:")
    end = next(
        i
        for i in range(block + 1, len(lines))
        if lines[i].startswith("    ") and not lines[i].startswith("      ")
    )

    header = (
        f"      # Produced by scripts/run_tuning.py on {date.today()}: "
        f"{cfg.tuning.n_trials} Optuna trials, seed {cfg.random_state}, CV PR-AUC {cv_mean:.6f}."
    )
    body = [f"      {key}: {_as_yaml(value)}" for key, value in params.items()]
    path.write_text(
        chr(10).join([*lines[: block + 1], header, *body, *lines[end:]]) + chr(10), encoding="utf-8"
    )
