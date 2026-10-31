"""Load the dataset and split it into training and test sets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import Config, load_config


@dataclass(frozen=True)
class DataSplit:
    """The train/test partition, created once and shared by every later stage."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        """Feature columns, in the order the models expect them."""
        return self.X_train.columns.tolist()

    @property
    def imbalance_ratio(self) -> float:
        """Negative-to-positive ratio in the training set.

        This is the neutral value for ``scale_pos_weight`` and the anchor for
        the range the tuner searches around it.
        """
        return float((self.y_train == 0).sum() / (self.y_train == 1).sum())

    def training_frame(self) -> pd.DataFrame:
        """Return training features and target together, for exploratory analysis.

        There is deliberately no equivalent for the test set: exploratory
        findings become preprocessing decisions, so letting them see the test
        data would stop the final metrics from being an unbiased estimate.
        """
        frame = self.X_train.copy()
        frame[self.y_train.name] = self.y_train
        return frame


def load_dataset(config: Config | None = None) -> pd.DataFrame:
    """Read the raw CSV and drop the columns listed in the configuration."""
    cfg = config or load_config()
    path = cfg.paths.absolute(cfg.paths.raw_data)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle "
            "('Give Me Some Credit') and place the CSV at that location."
        )

    frame = pd.read_csv(path)
    # The source export carries an unnamed row counter that would otherwise be
    # treated as a predictive feature.
    frame = frame.drop(columns=cfg.data.drop_columns, errors="ignore")

    if cfg.data.target not in frame.columns:
        raise ValueError(f"Target column {cfg.data.target!r} is missing from {path.name}")

    return frame


def split_data(frame: pd.DataFrame, config: Config | None = None) -> DataSplit:
    """Split the dataset into training and test sets, keeping the class ratio.

    With a positive class of roughly 7%, an unstratified split leaves the two
    partitions with different default rates, and PR-AUC responds directly to
    the base rate.
    """
    cfg = config or load_config()

    features = frame.drop(columns=[cfg.data.target])
    target = frame[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg.data.test_size,
        random_state=cfg.random_state,
        stratify=target if cfg.data.stratify else None,
    )

    # Overlapping partitions would make every test metric optimistic, and the
    # failure would be silent, so it is checked rather than assumed.
    overlap = X_train.index.intersection(X_test.index)
    if len(overlap) > 0:
        raise RuntimeError(f"Train and test sets share {len(overlap)} rows")

    return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
