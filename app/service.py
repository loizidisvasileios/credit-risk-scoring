"""Scoring logic behind the API: the one place that knows about the model."""

from __future__ import annotations

from functools import cache
from math import log

import pandas as pd
import shap
from sklearn.base import BaseEstimator

from credit_scoring.config import load_config
from credit_scoring.explainability import build_explainer, explain, reason_codes

# Guards the log in the score conversion: a probability of exactly zero or one
# has infinite odds, and a boosted model can round to either.
PROBABILITY_LIMIT = 1e-6


@cache
def load_service() -> tuple[BaseEstimator, dict]:
    """Load the saved model and its metadata once per process."""
    from credit_scoring.training import load_model

    return load_model()


@cache
def _explainer() -> shap.TreeExplainer:
    """Return the shared SHAP explainer for the loaded model."""
    model, _ = load_service()
    return build_explainer(model)


def credit_score(probability: float) -> int:
    """Convert a default probability to the points scale lenders publish.

    Scores rise as risk falls, which is the opposite direction to the
    probability and the reason the conversion exists.
    """
    scoring = load_config().scoring
    bounded = min(max(probability, PROBABILITY_LIMIT), 1.0 - PROBABILITY_LIMIT)
    score = scoring.offset + scoring.factor * log((1.0 - bounded) / bounded)
    return round(min(max(score, scoring.min_score), scoring.max_score))


def score_applicant(applicant: dict) -> dict:
    """Return the probability, score, decision and reason codes for one applicant."""
    model, metadata = load_service()
    # Reindexing to the stored feature order protects against a caller sending
    # the right fields in the wrong sequence.
    frame = pd.DataFrame([applicant])[metadata["features"]]

    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(metadata["threshold"])
    rejected = probability >= threshold

    return {
        "probability": probability,
        "score": credit_score(probability),
        "decision": "reject" if rejected else "approve",
        "threshold": threshold,
        # An approved applicant has no reasons for a decline to explain.
        "reason_codes": _reasons(model, frame) if rejected else [],
    }


def _reasons(model: BaseEstimator, frame: pd.DataFrame) -> list[dict]:
    """Return the features that pushed this applicant above the cut-off."""
    explanation = explain(model, frame, _explainer())
    return reason_codes(explanation, 0).to_dict(orient="records")
