"""Pickle-to-image extraction for the GTSRB 4GB dataset.

The upstream 4GB Kaggle bundle ships nine pickle files (``data0.pickle`` ..
``data8.pickle``) each containing preprocessed RGB or grayscale tensors.
FastAI's ``DataBlock`` needs files on disk, so this module walks a pickle and
writes out ``label/image_i.png`` into a ``GrandparentSplitter``-friendly layout.

The original name was ``proecssing.py`` (typo in the class project); that has
been fixed here.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dashcam_sign_detector.classifier.config import ClassifierConfig


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def save_images_to_disk(
    images: np.ndarray,
    labels: np.ndarray,
    base_path: Path,
) -> int:
    """Write ``images`` into ``base_path/<label>/image_<idx>.png``.

    Handles both (1, H, W) grayscale and (3, H, W) RGB tensors. Returns the
    number of images written.
    """
    base_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        if image.shape[0] == 1:
            arr = image.squeeze(0)
            mode = "L"
        else:
            arr = image.transpose(1, 2, 0)
            mode = "RGB"

        img = Image.fromarray(arr.astype("uint8"), mode)
        label_dir = base_path / str(int(label))
        label_dir.mkdir(parents=True, exist_ok=True)
        img.save(label_dir / f"image_{i}.png")
        count += 1
    return count


def extract_split(cfg: ClassifierConfig) -> None:
    """Extract the pickle for ``cfg.data_id`` into the processed image tree."""
    pickle_path = cfg.pickle_path
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Expected preprocessed pickle at {pickle_path}. "
            "Run scripts/download_data.py first."
        )

    data = load_pickle(pickle_path)
    out_root = cfg.images_dir
    out_root.mkdir(parents=True, exist_ok=True)

    splits = {
        cfg.train_split: ("x_train", "y_train"),
        cfg.valid_split: ("x_validation", "y_validation"),
        cfg.test_split: ("x_test", "y_test"),
    }
    for split_name, (xk, yk) in splits.items():
        written = save_images_to_disk(data[xk], data[yk], out_root / split_name)
        print(f"  {split_name}: wrote {written} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GTSRB pickle into images on disk.")
    parser.add_argument("--data-id", type=int, default=None, help="Pickle id 0..8.")
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.data_id is not None:
        cfg.data_id = args.data_id
    cfg.ensure_dirs()

    print(f"Extracting data{cfg.data_id}.pickle -> {cfg.images_dir}")
    extract_split(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
