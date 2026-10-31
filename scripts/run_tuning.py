"""Search hyperparameters with Optuna and record the result.

Usage:
    python scripts/run_tuning.py                 # every configured model
    python scripts/run_tuning.py xgboost         # one model
    python scripts/run_tuning.py --keep-config   # search but leave config.yaml alone
"""

from __future__ import annotations

import argparse
import time

from credit_scoring.config import load_config
from credit_scoring.data import load_dataset, split_data
from credit_scoring.tuning import run_study, save_trials, write_best_params


def main() -> None:
    """Run one study per requested model, saving its trials and best parameters."""
    cfg = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=None, help="models to tune")
    parser.add_argument(
        "--keep-config", action="store_true", help="do not write the tuned values back"
    )
    args = parser.parse_args()

    names = args.models or list(cfg.models)
    unknown = [name for name in names if name not in cfg.models]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Available: {list(cfg.models)}")

    split = split_data(load_dataset(cfg), cfg)
    print(
        f"Tuning {len(names)} model(s), {cfg.tuning.n_trials} trials each, seed {cfg.random_state}"
    )

    for name in names:
        started = time.perf_counter()
        study = run_study(name, split.X_train, split.y_train, cfg)
        path = save_trials(study, cfg)

        print(f"\n{name}  [{(time.perf_counter() - started) / 60:.1f} min]")
        print(f"  best {cfg.cross_validation.scoring}: {study.best_value:.6f}")
        print(f"  trials saved to: {path}")

        if args.keep_config:
            print("  config.yaml left unchanged (--keep-config)")
        else:
            write_best_params(name, study.best_params, study.best_value, cfg)
            print("  best parameters written to config.yaml")


if __name__ == "__main__":
    main()
