"""Tests for the scoring API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app
from credit_scoring.config import load_config

client = TestClient(app)

# A borderline applicant from the test set: moderate utilisation, three short
# delinquencies, median income.
APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.362,
    "age": 55,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.648,
    "MonthlyIncome": 3325,
    "NumberOfOpenCreditLinesAndLoans": 12,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}

HIGH_RISK = {**APPLICANT, "NumberOfTimes90DaysLate": 3, "NumberOfTime60-89DaysPastDueNotWorse": 2}
LOW_RISK = {**APPLICANT, "NumberOfTime30-59DaysPastDueNotWorse": 0, "MonthlyIncome": 12000}


def test_health_reports_a_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_valid_applicant_is_scored():
    response = client.post("/predict", json=APPLICANT)
    body = response.json()
    scoring = load_config().scoring

    assert response.status_code == 200
    assert 0.0 <= body["probability"] <= 1.0
    assert scoring.min_score <= body["score"] <= scoring.max_score


@pytest.mark.parametrize(
    ("field", "value"),
    [("age", 5), ("NumberOfTimes90DaysLate", 98), ("MonthlyIncome", -100)],
)
def test_impossible_values_are_rejected(field, value):
    """Invalid applications must fail validation instead of reaching the model."""
    response = client.post("/predict", json={**APPLICANT, field: value})

    assert response.status_code == 422


def test_declined_applicant_gets_reasons():
    body = client.post("/predict", json=HIGH_RISK).json()

    assert body["decision"] == "reject"
    assert body["probability"] >= body["threshold"]
    assert len(body["reason_codes"]) == 3


def test_approved_applicant_gets_no_reasons():
    """Reason codes explain a decline, so an approval must not carry any."""
    body = client.post("/predict", json=LOW_RISK).json()

    assert body["decision"] == "approve"
    assert body["probability"] < body["threshold"]
    assert body["reason_codes"] == []
