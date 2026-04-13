"""Download the GTSRB 4GB preprocessed bundle from Kaggle.

Prereqs (see ``data/README.md``):

    uv pip install kaggle  # already in the project deps
    # Drop a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)

Usage:

    uv run python scripts/download_data.py

The bundle lands in ``$DSD_DATA_ROOT/raw/gtsrb_4gb`` (default:
``<repo>/data/raw/gtsrb_4gb``). The raw data tree is gitignored.

After download, run the pickle-to-image extractor:

    uv run python -m dashcam_sign_detector.classifier.preprocessing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from dashcam_sign_detector.classifier.config import ClassifierConfig

# Upstream Kaggle dataset slug for the 4GB preprocessed bundle.
KAGGLE_DATASET = "valentynsichkar/traffic-signs-preprocessed"


def _require_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        print(
            "error: `kaggle` CLI not found on PATH. Install project deps with "
            "`uv sync`, then ensure ~/.kaggle/kaggle.json exists.",
            file=sys.stderr,
        )
        sys.exit(1)


def download(cfg: ClassifierConfig, force: bool = False) -> None:
    _require_kaggle_cli()
    target = cfg.raw_dir
    target.mkdir(parents=True, exist_ok=True)

    if any(target.glob("*.pickle")) and not force:
        print(f"Found existing pickle files in {target}, skipping download.")
        print("Pass --force to redownload.")
        return

    print(f"Downloading {KAGGLE_DATASET} -> {target}")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(target),
            "--unzip",
        ],
        check=True,
    )
    print("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GTSRB 4GB preprocessed bundle.")
    parser.add_argument("--force", action="store_true", help="Redownload even if files exist.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override DSD_DATA_ROOT.")
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root.expanduser().resolve()
    cfg.ensure_dirs()
    download(cfg, force=args.force)


if __name__ == "__main__":
    main()
