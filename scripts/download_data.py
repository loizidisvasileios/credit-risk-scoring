"""Fetch the Give Me Some Credit dataset from Kaggle.

The competition data cannot be downloaded anonymously: a Kaggle account must
have accepted the competition rules, and an API token must sit at
~/.kaggle/kaggle.json. Without both, download the file by hand from
https://www.kaggle.com/c/GiveMeSomeCredit/data and place it at the path the
configuration names.

Usage:
    pip install -e ".[data]"
    python scripts/download_data.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from credit_scoring.config import load_config

COMPETITION = "GiveMeSomeCredit"


def download(destination: Path) -> None:
    """Download and unpack the competition archive into ``destination``."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise SystemExit(
            'The kaggle package is not installed. Run: pip install -e ".[data]"'
        ) from error

    api = KaggleApi()
    # Reads ~/.kaggle/kaggle.json and fails here if the token is missing, which
    # is a clearer place to stop than halfway through a download.
    api.authenticate()
    api.competition_download_files(COMPETITION, path=str(destination))

    archive = destination / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()


def main() -> None:
    """Place the raw dataset where the configuration expects it."""
    cfg = load_config()
    target = cfg.paths.absolute(cfg.paths.raw_data)

    if target.is_file():
        print(f"Already present: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} to {target.parent}")
    download(target.parent)

    if not target.is_file():
        raise SystemExit(f"Download finished but {target.name} is missing from the archive.")
    print(f"Saved: {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
