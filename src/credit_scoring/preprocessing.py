"""Preprocessing steps, built per model and fitted inside the pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from credit_scoring.config import Config, ModelConfig, load_config


def _require(X: object, columns: Sequence[str], owner: str) -> pd.DataFrame:
    """Return ``X`` as a DataFrame with the expected columns, or raise."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{owner} needs a DataFrame, received {type(X).__name__}")
    missing = [column for column in columns if column not in X.columns]
    if missing:
        raise ValueError(f"{owner} expects columns that are absent: {missing}")
    return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Cap extreme values at quantiles learnt from the data seen in ``fit``.

    Inside cross-validation each fold therefore derives its own limits from its
    own training part, so nothing is learnt from the data it is scored on.
    """

    def __init__(
        self,
        winsorize_columns: Sequence[str] = (),
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        age_column: str | None = None,
        age_bounds: tuple[int, int] = (18, 120),
    ) -> None:
        self.winsorize_columns = winsorize_columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.age_column = age_column
        self.age_bounds = age_bounds

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> OutlierClipper:
        """Learn the clipping limits."""
        frame = _require(X, self.winsorize_columns, "OutlierClipper")
        self.bounds_ = {
            column: (
                float(frame[column].quantile(self.lower_quantile)),
                float(frame[column].quantile(self.upper_quantile)),
            )
            for column in self.winsorize_columns
        }
        self.feature_names_in_ = frame.columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learnt limits to a copy of ``X``."""
        check_is_fitted(self, "bounds_")
        # A copy, because modifying the caller's frame would let one fold's
        # clipping carry over into the next.
        frame = _require(X, self.winsorize_columns, "OutlierClipper").copy()

        for column, (lower, upper) in self.bounds_.items():
            frame[column] = frame[column].clip(lower=lower, upper=upper)

        if self.age_column is not None:
            frame[self.age_column] = frame[self.age_column].clip(*self.age_bounds)

        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Clipping changes values, not columns, so the names are unchanged."""
        check_is_fitted(self, "bounds_")
        return list(input_features) if input_features is not None else list(self.feature_names_in_)


class DelinquencyCombiner(BaseEstimator, TransformerMixin):
    """Replace the three delinquency counters with their total.

    They correlate above 0.98 with one another, which leaves linear
    coefficients unstable because the fit can move weight between them almost
    freely. Tree ensembles do not have that problem and keep the raw counters.
    """

    def __init__(
        self, columns: Sequence[str] = (), output_column: str = "TotalLatePayments"
    ) -> None:
        self.columns = columns
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DelinquencyCombiner:
        """Record the resulting column order."""
        frame = _require(X, self.columns, "DelinquencyCombiner")
        kept = [column for column in frame.columns if column not in self.columns]
        self.feature_names_out_ = [*kept, self.output_column]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sum the counters, drop them, and return a fixed column order."""
        check_is_fitted(self, "feature_names_out_")
        frame = _require(X, self.columns, "DelinquencyCombiner").copy()
        frame[self.output_column] = frame[list(self.columns)].sum(axis=1)
        # The explicit ordering matters: if columns arrived in a different order
        # at predict time, the model would read the wrong feature in silence.
        return frame.drop(columns=list(self.columns))[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return the column names after the counters are merged."""
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)


def preprocessing_steps(
    model_config: ModelConfig, config: Config | None = None
) -> list[tuple[str, BaseEstimator]]:
    """Return the pipeline steps this model needs, in the order they apply.

    The three flags come from the configuration and record why each model is
    treated differently: trees split on the order of values, so scaling and
    clipping change nothing for them, and they absorb correlated inputs.
    """
    cfg = config or load_config()
    pre = cfg.preprocessing
    steps: list[tuple[str, BaseEstimator]] = []

    if model_config.clip_outliers:
        steps.append(
            (
                "clipper",
                OutlierClipper(
                    winsorize_columns=pre.winsorize.columns,
                    lower_quantile=pre.winsorize.lower_quantile,
                    upper_quantile=pre.winsorize.upper_quantile,
                    age_column=pre.age_column,
                    age_bounds=(pre.age_bounds.min, pre.age_bounds.max),
                ),
            )
        )

    if model_config.combine_delinquency:
        steps.append(
            (
                "delinquency",
                DelinquencyCombiner(
                    columns=pre.delinquency.columns,
                    output_column=pre.delinquency.output_column,
                ),
            )
        )

    # pandas output keeps column names alive through the chain, which the SHAP
    # step later needs to label its explanations.
    steps.append(
        ("imputer", SimpleImputer(strategy=pre.imputation_strategy).set_output(transform="pandas"))
    )

    if model_config.scale_features:
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))

    return steps
