"""GTSRB data loading for FastAI.

Builds two ``DataLoaders``: one unaugmented (phase 1a) and one with selective
augmentation (phase 1b). ``labels_to_avoid_rotation`` covers the directional
signs (Go straight or {left,right}, Keep {left,right}) which previously
collapsed into each other when rotated during augmentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    GrandparentSplitter,
    ImageBlock,
    PILImage,
    Resize,
    aug_transforms,
    get_image_files,
)
from fastai.vision.augment import RandTransform

from dashcam_sign_detector.classifier.config import ClassifierConfig


class SelectiveRotation(RandTransform):
    """Rotate images *except* those belonging to the given label directories."""

    def __init__(self, labels_to_avoid: tuple[str, ...], degrees: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.labels_to_avoid = set(labels_to_avoid)
        self.degrees = degrees

    def encodes(self, x: PILImage) -> PILImage:
        label_name = x.parent.name if hasattr(x, "parent") else None
        if label_name in self.labels_to_avoid:
            return x
        degree = random.uniform(-self.degrees, self.degrees)
        return x.rotate(degree)


def _label_from_path(item: Path) -> str:
    return item.parent.name


def _all_labels(images_dir: Path) -> list[str]:
    return sorted({p.parent.name for p in get_image_files(images_dir)})


@dataclass
class ClassifierData:
    reg_dls: DataLoaders
    aug_dls: DataLoaders
    test_dl: DataLoaders
    label_names: list[str]


def build_dataloaders(cfg: ClassifierConfig) -> ClassifierData:
    """Build unaugmented + augmented DataLoaders and a held-out test DL."""
    images_dir = cfg.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Expected processed images at {images_dir}. "
            "Run `python -m dashcam_sign_detector.classifier.preprocessing` first."
        )

    labels = _all_labels(images_dir)

    reg_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=labels)),
        get_items=get_image_files,
        splitter=GrandparentSplitter(
            train_name=cfg.train_split, valid_name=cfg.valid_split
        ),
        get_y=_label_from_path,
        item_tfms=Resize(cfg.image_size),
    )

    aug_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=labels)),
        get_items=get_image_files,
        splitter=GrandparentSplitter(
            train_name=cfg.train_split, valid_name=cfg.valid_split
        ),
        get_y=_label_from_path,
        item_tfms=Resize(cfg.image_size),
        batch_tfms=[
            *aug_transforms(mult=1.0, do_flip=True, flip_vert=False, max_warp=0.0),
            SelectiveRotation(cfg.labels_to_avoid_rotation, degrees=10.0),
        ],
    )

    reg_dls = reg_block.dataloaders(
        images_dir, bs=cfg.batch_size, num_workers=cfg.num_workers
    )
    aug_dls = aug_block.dataloaders(
        images_dir, bs=cfg.batch_size, num_workers=cfg.num_workers
    )

    test_items = get_image_files(images_dir / cfg.test_split)
    test_dl = reg_dls.test_dl(test_items, with_labels=True)

    return ClassifierData(
        reg_dls=reg_dls,
        aug_dls=aug_dls,
        test_dl=test_dl,
        label_names=labels,
    )
