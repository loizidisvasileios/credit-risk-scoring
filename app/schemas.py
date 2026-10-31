"""Request and response schemas for the scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The source dataset uses 96 and 98 as codes rather than counts, so anything
# above the highest real count is rejected before it reaches the model.
MAX_LATE_COUNT = 20


class Applicant(BaseModel):
    """One credit application.

    Field names follow Python conventions while the aliases keep the JSON
    payload identical to the dataset columns, so a request can be copied
    straight out of the data.
    """

    model_config = ConfigDict(populate_by_name=True)

    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines", ge=0.0, le=100.0
    )
    age: int = Field(ge=18, le=120)
    late_30_59_days: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le=MAX_LATE_COUNT
    )
    debt_ratio: float = Field(alias="DebtRatio", ge=0.0)
    # Missing income is common in the data and handled by the pipeline's
    # imputer, so the field is optional rather than required.
    monthly_income: float | None = Field(default=None, alias="MonthlyIncome", ge=0.0)
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0, le=100)
    late_90_days: int = Field(alias="NumberOfTimes90DaysLate", ge=0, le=MAX_LATE_COUNT)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0, le=100)
    late_60_89_days: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le=MAX_LATE_COUNT
    )
    dependents: int | None = Field(default=None, alias="NumberOfDependents", ge=0, le=50)


class ReasonCode(BaseModel):
    """One feature's contribution to a decline."""

    feature: str
    value: float
    contribution: float


class Decision(BaseModel):
    """The scoring result returned for an application."""

    probability: float = Field(ge=0.0, le=1.0)
    score: int
    decision: Literal["approve", "reject"]
    threshold: float
    reason_codes: list[ReasonCode]


class Health(BaseModel):
    """Service state, used by deployment health checks."""

    status: Literal["ok"]
    model_loaded: bool
