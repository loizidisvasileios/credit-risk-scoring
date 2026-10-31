"""Train the final model and move it to and from disk."""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from credit_scoring.config import Config, load_config
from credit_scoring.models import build_pipeline

MODEL_FILE = "credit_model.pkl"
METADATA_FILE = "model_metadata.json"


def train_model(
    name: str, X: pd.DataFrame, y: pd.Series, config: Config | None = None
) -> BaseEstimator:
    """Fit ``name`` on the whole training set with its tuned parameters.

    Cross-validation fits models on four fifths of the data and throws them
    away; this is the one fit that keeps everything and becomes the model that
    is actually used.
    """
    cfg = config or load_config()
    return build_pipeline(name, cfg).fit(X, y)


def save_model(
    model: BaseEstimator,
    name: str,
    features: list[str],
    config: Config | None = None,
    filename: str = MODEL_FILE,
) -> tuple[Path, Path]:
    """Write the model next to a metadata file and return both paths.

    The metadata carries what a caller needs but cannot read off the pickle:
    the feature order the model expects, the decision threshold with the
    assumptions behind it, and the library versions, because a model pickled by
    one version of scikit-learn or XGBoost may not load under another.
    """
    cfg = config or load_config()
    directory = cfg.paths.absolute(cfg.paths.models_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / filename
    joblib.dump(model, model_path)

    metadata = {
        "model": name,
        # Read off the object itself, so the file can never claim a calibration
        # the saved model does not have.
        "calibrated": isinstance(model, CalibratedClassifierCV),
        "features": features,
        "threshold": cfg.threshold.optimal_cutoff,
        "threshold_derivation": {
            "rule": "tau = 1 / (k + 1), k = loss_given_default / net_margin",
            "loss_given_default": cfg.threshold.loss_given_default,
            "net_margin": cfg.threshold.net_margin,
            "cost_ratio": cfg.threshold.cost_ratio,
        },
        "trained_at": str(date.today()),
        "random_state": cfg.random_state,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }

    metadata_path = directory / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_model(
    config: Config | None = None, filename: str = MODEL_FILE
) -> tuple[BaseEstimator, dict]:
    """Load the saved model together with its metadata."""
    cfg = config or load_config()
    directory = cfg.paths.absolute(cfg.paths.models_dir)
    model_path = directory / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"No model at {model_path}. Run 'python scripts/run_training.py' first."
        )

    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata
