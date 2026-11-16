"""Train the final model on the training set and save it.

Usage:
    python scripts/run_training.py
"""

from __future__ import annotations

from credit_scoring.calibration import calibrate_model
from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.metrics import compute_metrics
from credit_scoring.training import save_explanation_model, save_model, train_model


def main() -> None:
    """Fit the chosen model on every training row and write it to the models directory."""
    cfg = load_config()

    name = cfg.selected_model
    split = split_data(load_dataset(cfg), cfg)

    print(f"Training {name} on {len(split.X_train):,} rows, seed {cfg.random_state}")
    print(f"Calibrating with {cfg.calibration.method} regression, cv={cfg.calibration.cv}")
    model = calibrate_model(split.X_train, split.y_train, cfg)

    explanation_model = train_model(name, split.X_train, split.y_train, cfg)

    proba = model.predict_proba(split.X_test)[:, 1]
    metrics = compute_metrics(split.y_test, proba)
    print("\ntest metrics")
    for metric, value in metrics.items():
        print(f"  {metric:<8} {value:.4f}")


    print(f"\nmean predicted probability: {proba.mean():.4f}")
    print(f"observed default rate     : {split.y_test.mean():.4f}")

    model_path, metadata_path = save_model(model, name, split.feature_names, cfg)
    explanation_path = save_explanation_model(explanation_model, cfg)
    print(f"\nsaved: {model_path}")
    print(f"saved: {metadata_path}")
    print(f"saved: {explanation_path}")


if __name__ == "__main__":
    main()
