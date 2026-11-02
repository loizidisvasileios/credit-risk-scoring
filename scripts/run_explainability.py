"""Explain the saved model with SHAP, globally and for individual applicants.

Usage:
    python scripts/run_explainability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.explainability import (
    explain,
    plot_beeswarm,
    plot_importance,
    plot_waterfall,
    reason_codes,
)
from credit_scoring.training import load_explanation_model, load_model

TABLE_FILE = "reason_codes.csv"


def main() -> None:
    """Produce the two global figures, then one breakdown per selected applicant."""
    cfg = load_config()
    model, metadata = load_model(cfg)
    explanation_model = load_explanation_model(cfg)
    split = split_data(load_dataset(cfg), cfg)

    sample = split.X_test.sample(n=cfg.explainability.sample_size, random_state=cfg.random_state)
    
    proba = model.predict_proba(sample)[:, 1]
    print(f"Explaining {len(sample):,} test applicants")

    explanation = explain(explanation_model, sample)
    plot_importance(explanation, cfg)
    plot_beeswarm(explanation, cfg)

    threshold = metadata["threshold"]
    
    selected = {
        "highest_risk": int(proba.argmax()),
        "lowest_risk": int(proba.argmin()),
        "at_threshold": int(np.abs(proba - threshold).argmin()),
    }

    tables = []
    for label, index in selected.items():
        plot_waterfall(explanation, index, label, cfg)
        codes = reason_codes(explanation, index)

        print(f"\n{label}  probability {proba[index]:.4f}  threshold {threshold:.4f}")
        print(codes.round(4).to_string(index=False))
        tables.append(codes.assign(applicant=label, probability=proba[index]))

    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(directory / TABLE_FILE, index=False)
    print(f"\nsaved: {directory / TABLE_FILE}")


if __name__ == "__main__":
    main()
