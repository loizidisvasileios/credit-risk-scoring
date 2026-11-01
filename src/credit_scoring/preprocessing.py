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
    """Validate that ``X`` is a DataFrame and contains the columns this step needs."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{owner} needs a DataFrame, received {type(X).__name__}")
    missing = [column for column in columns if column not in X.columns]
    if missing:
        raise ValueError(f"{owner} expects columns that are absent: {missing}")
    return X


class DataQualityCleaner(BaseEstimator, TransformerMixin):
    """Turn recorded values that are not measurements into missing values, and flag them."""

    def __init__(
        self,
        delinquency_columns: Sequence[str] = (),
        delinquency_codes: Sequence[int] = (),
        utilisation_column: str | None = None,
        utilisation_max: float = float("inf"),
        ratio_column: str | None = None,
        income_column: str | None = None,
        flag_missing_columns: Sequence[str] = (),
        delinquency_flag: str = "DelinquencyCodeFlag",
    ) -> None:
        self.delinquency_columns = delinquency_columns
        self.delinquency_codes = delinquency_codes
        self.utilisation_column = utilisation_column
        self.utilisation_max = utilisation_max
        self.ratio_column = ratio_column
        self.income_column = income_column
        self.flag_missing_columns = flag_missing_columns
        self.delinquency_flag = delinquency_flag

    def _flag_names(self) -> list[str]:
        """Name every flag this transformer appends, in the order it appends them."""
        names = []
        if self.delinquency_columns:
            
            names.append(self.delinquency_flag)
        if self.utilisation_column:
            names.append(f"{self.utilisation_column}Invalid")
        names += [f"{column}Missing" for column in self.flag_missing_columns]
        return names

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DataQualityCleaner:
        """Record the column order the transform produces."""
        frame = _require(X, [], "DataQualityCleaner")
        self.feature_names_out_ = [*frame.columns, *self._flag_names()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Blank out the unusable values and append one flag per rule."""
        check_is_fitted(self, "feature_names_out_")
        frame = _require(X, [], "DataQualityCleaner").copy()

        if self.delinquency_columns:
            columns = list(self.delinquency_columns)
            coded = frame[columns].isin(list(self.delinquency_codes))
            frame[self.delinquency_flag] = coded.any(axis=1).astype(int)
            frame[columns] = frame[columns].mask(coded)

        if self.utilisation_column:
            invalid = frame[self.utilisation_column] > self.utilisation_max
            frame[f"{self.utilisation_column}Invalid"] = invalid.astype(int)
            frame[self.utilisation_column] = frame[self.utilisation_column].mask(invalid)

        if self.ratio_column and self.income_column:
          
            frame[self.ratio_column] = frame[self.ratio_column].mask(
                frame[self.income_column].isna()
            )

        for column in self.flag_missing_columns:
            frame[f"{column}Missing"] = frame[column].isna().astype(int)

        return frame[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return the original columns followed by the appended flags."""
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip the extreme values of the columns that are winsorised."""

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
        """Transform the frame by clipping the columns to the fitted limits."""
        check_is_fitted(self, "bounds_")
        
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
    """Replace the three delinquency counters with their total."""

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
        
        frame[self.output_column] = frame[list(self.columns)].sum(axis=1, min_count=1)
        
        return frame.drop(columns=list(self.columns))[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return the column names after the counters are merged."""
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)


def preprocessing_steps(
    model_config: ModelConfig, config: Config | None = None
) -> list[tuple[str, BaseEstimator]]:
    """Return the pipeline steps this model needs, in the order they apply."""
    cfg = config or load_config()
    pre = cfg.preprocessing
    quality = cfg.data_quality


    steps: list[tuple[str, BaseEstimator]] = [
        (
            "quality",
            DataQualityCleaner(
                delinquency_columns=pre.delinquency.columns,
                delinquency_codes=quality.delinquency_codes,
                utilisation_column=quality.utilisation_column,
                utilisation_max=quality.utilisation_max,
                ratio_column=quality.ratio_column,
                income_column=quality.income_column,
                flag_missing_columns=quality.flag_missing_columns,
            ),
        )
    ]

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

    
    steps.append(
        ("imputer", SimpleImputer(strategy=pre.imputation_strategy).set_output(transform="pandas"))
    )

    if model_config.scale_features:
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))

    return steps
