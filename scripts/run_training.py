"""Train the final model on the full training set and save it for serving.

Usage:
    python scripts/run_training.py
"""

from __future__ import annotations

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compute_metrics
from credit_scoring.training import save_model, train_model

FINAL_MODEL = "xgboost"


def main() -> None:
    """Fit the chosen model on every training row and write it to the models directory."""
    cfg = load_config()
    split = split_data(load_dataset(cfg), cfg)

    print(f"Training {FINAL_MODEL} on {len(split.X_train):,} rows, seed {cfg.random_state}")
    model = train_model(FINAL_MODEL, split.X_train, split.y_train, cfg)

    proba = model.predict_proba(split.X_test)[:, 1]
    metrics = compute_metrics(split.y_test, proba)
    print("\ntest metrics")
    for metric, value in metrics.items():
        print(f"  {metric:<8} {value:.4f}")

    # The threshold is only meaningful if predicted probabilities sit at the
    # right level, so the two averages are printed together as a sanity check.
    print(f"\nmean predicted probability: {proba.mean():.4f}")
    print(f"observed default rate     : {split.y_test.mean():.4f}")

    model_path, metadata_path = save_model(model, FINAL_MODEL, split.feature_names, cfg)
    print(f"\nsaved: {model_path}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
