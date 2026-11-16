"""REST API that serves credit decisions.

Usage:
    uvicorn app.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from app.schemas import Applicant, Decision, Health
from app.service import load_service, score_applicant

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Default probability, credit score and decline reasons for a credit application.",
    version="1.0.0",
)


@app.get("/health", response_model=Health)
def health() -> Health:
    """Report whether the service is up and the model is loaded."""
    model, _ = load_service()
    return Health(status="ok", model_loaded=model is not None)


@app.get("/model")
def model_info() -> dict:
    """Return the metadata saved alongside the model."""
    _, metadata = load_service()
    return metadata


@app.post("/predict", response_model=Decision)
def predict(applicant: Applicant) -> Decision:
    """Score one application and return the decision behind it."""
    return Decision(**score_applicant(applicant.model_dump(by_alias=True)))
