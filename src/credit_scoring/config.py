"""Load and validate the project configuration."""

from __future__ import annotations

from functools import cache
from math import log
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The package sits at <root>/src/credit_scoring, so the repository root is two
# levels up. Deriving it here lets scripts run from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class _Section(BaseModel):
    """Base for every config section: unknown keys are errors, values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectInfo(_Section):
    """Name and description used in reports."""

    name: str
    description: str


class Paths(_Section):
    """Locations of the dataset and of generated artefacts."""

    raw_data: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    results_dir: Path

    def absolute(self, path: Path) -> Path:
        """Return ``path`` anchored at the repository root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class DataConfig(_Section):
    """Target column and train/test split policy."""

    target: str
    drop_columns: list[str]
    test_size: float = Field(gt=0.0, lt=1.0)
    stratify: bool


class WinsorizeConfig(_Section):
    """Quantiles at which heavy-tailed features are capped."""

    columns: list[str]
    lower_quantile: float = Field(ge=0.0, lt=1.0)
    upper_quantile: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> WinsorizeConfig:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                f"lower_quantile ({self.lower_quantile}) must be below "
                f"upper_quantile ({self.upper_quantile})"
            )
        return self


class AgeBounds(_Section):
    """Plausible age range, set by domain knowledge rather than by the data."""

    min: int = Field(ge=0)
    max: int = Field(gt=0)


class DelinquencyConfig(_Section):
    """The three near-collinear delinquency counters and their aggregate."""

    columns: list[str]
    output_column: str


class PreprocessingConfig(_Section):
    """Cleaning and feature engineering settings."""

    winsorize: WinsorizeConfig
    age_column: str
    age_bounds: AgeBounds
    delinquency: DelinquencyConfig
    imputation_strategy: Literal["median", "mean", "most_frequent"]


class CrossValidationConfig(_Section):
    """Resampling protocol shared by tuning and model comparison."""

    n_splits: int = Field(ge=2)
    shuffle: bool
    scoring: str


class TuningConfig(_Section):
    """Search budget for Optuna."""

    n_trials: int = Field(ge=1)
    direction: Literal["maximize", "minimize"]


class ModelConfig(_Section):
    """One estimator: the preprocessing it needs and its parameters."""

    display_name: str
    scale_features: bool
    combine_delinquency: bool
    clip_outliers: bool
    fixed_params: dict[str, Any] = Field(default_factory=dict)
    best_params: dict[str, Any]
    search_space: dict[str, dict[str, Any]]

    def params(self) -> dict[str, Any]:
        """Return the keyword arguments used to build the estimator."""
        return {**self.fixed_params, **self.best_params}


class CalibrationConfig(_Section):
    """Probability calibration settings for the production model."""

    model: str
    method: Literal["isotonic", "sigmoid"]
    cv: int = Field(ge=2)
    n_bins: int = Field(ge=2)
    bin_strategy: Literal["quantile", "uniform"]


class ThresholdConfig(_Section):
    """Economic assumptions behind the accept/reject cut-off.

    Rejecting a good applicant forgoes the margin ``r``; accepting a defaulting
    one loses ``LGD`` of the exposure. With ``k = LGD / r``, the cost-optimal
    cut-off on a calibrated probability is ``1 / (k + 1)``.
    """

    loss_given_default: float = Field(gt=0.0, le=1.0)
    net_margin: float = Field(gt=0.0, le=1.0)
    sensitivity_k_grid: list[float]

    @property
    def cost_ratio(self) -> float:
        """Cost of a false negative relative to a false positive."""
        return self.loss_given_default / self.net_margin

    @property
    def optimal_cutoff(self) -> float:
        """Probability above which rejecting is the cheaper decision."""
        return 1.0 / (self.cost_ratio + 1.0)


class ExplainabilityConfig(_Section):
    """SHAP sampling and display settings."""

    sample_size: int = Field(ge=1)
    max_display: int = Field(ge=1)


class ScoringConfig(_Section):
    """Points scale the served probability is reported on.

    A probability is the right output for a decision rule but an awkward one to
    read, so lenders publish a score instead: it rises as risk falls, and every
    ``points_to_double`` points doubles the odds of repayment.
    """

    base_score: int = Field(ge=0)
    base_odds: float = Field(gt=0.0)
    points_to_double: int = Field(gt=0)
    min_score: int = Field(ge=0)
    max_score: int = Field(ge=0)

    @property
    def factor(self) -> float:
        """Points added for each unit increase in the log odds of repayment."""
        return self.points_to_double / log(2.0)

    @property
    def offset(self) -> float:
        """Points that place ``base_odds`` exactly at ``base_score``."""
        return self.base_score - self.factor * log(self.base_odds)


class PlottingConfig(_Section):
    """Figure rendering defaults."""

    dpi: int = Field(ge=72)
    format: str
    style: str
    figsize_default: tuple[float, float]


class Config(_Section):
    """The complete, validated configuration."""

    project: ProjectInfo
    random_state: int
    paths: Paths
    data: DataConfig
    preprocessing: PreprocessingConfig
    cross_validation: CrossValidationConfig
    tuning: TuningConfig
    models: dict[str, ModelConfig]
    calibration: CalibrationConfig
    threshold: ThresholdConfig
    explainability: ExplainabilityConfig
    scoring: ScoringConfig
    plotting: PlottingConfig

    @model_validator(mode="after")
    def _check_calibrated_model_exists(self) -> Config:
        if self.calibration.model not in self.models:
            raise ValueError(
                f"calibration.model is {self.calibration.model!r}, which is not one "
                f"of the defined models: {sorted(self.models)}"
            )
        return self


@cache
def load_config(config_path: str | Path | None = None) -> Config:
    """Read and validate the YAML configuration, caching the result per path."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return Config.model_validate(raw)
