"""Three-phase training schedule for the GTSRB classifier.

This reproduces the hand-tuned FastAI ``fit_one_cycle`` recipe that gave the
original TSC project ~97% test accuracy:

    Phase 1a — frozen head, unaugmented data, ``phase1_epochs_clean`` epochs
    Phase 1b — frozen head, selective-rotation augmentation, ``phase1_epochs_augmented`` epochs
    Phase 2  — unfrozen, differential LR slice(low, high), ``phase2_epochs`` epochs

The exact LR numbers in ``ClassifierConfig`` come from the original ``lr_find``
runs. They are reasonable defaults; pass ``--lr-find`` to re-run ``lr_find``
and print fresh suggestions before training.
"""

from __future__ import annotations

import argparse

from fastai.vision.all import Learner

from dashcam_sign_detector.classifier.config import ClassifierConfig
from dashcam_sign_detector.classifier.dataset import (
    ClassifierData,
    build_dataloaders,
)
from dashcam_sign_detector.classifier.model import build_learner


def _print_lr_suggestion(learn: Learner, label: str) -> None:
    suggestions = learn.lr_find()
    print(f"[{label}] lr_find suggestion: {suggestions}")


def train(cfg: ClassifierConfig, run_lr_find: bool = False) -> tuple[Learner, ClassifierData]:
    cfg.ensure_dirs()

    print(f"Building dataloaders from {cfg.images_dir}")
    data = build_dataloaders(cfg)
    n_classes = len(data.label_names)
    print(f"Found {n_classes} classes: {data.label_names[:5]}...")

    print(f"Building learner with backbone={cfg.backbone} pretrained=True")
    learn = build_learner(cfg, data.reg_dls, n_classes)

    if run_lr_find:
        _print_lr_suggestion(learn, "phase1 frozen")

    print(f"Phase 1a: frozen + unaugmented, {cfg.phase1_epochs_clean} epochs @ lr={cfg.phase1_lr}")
    learn.fit_one_cycle(cfg.phase1_epochs_clean, lr_max=cfg.phase1_lr)

    print(f"Phase 1b: frozen + augmented, {cfg.phase1_epochs_augmented} epochs @ lr={cfg.phase1_lr}")
    learn.dls = data.aug_dls
    learn.fit_one_cycle(cfg.phase1_epochs_augmented, lr_max=cfg.phase1_lr)

    print("Phase 2: unfreeze + differential LR")
    learn.unfreeze()
    if run_lr_find:
        _print_lr_suggestion(learn, "phase2 unfrozen")

    lr_slice = slice(cfg.phase2_lr_low, cfg.phase2_lr_high)
    print(f"Phase 2: {cfg.phase2_epochs} epochs @ lr={lr_slice}")
    learn.fit_one_cycle(cfg.phase2_epochs, lr_max=lr_slice)

    model_path = cfg.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learn.export(model_path)
    print(f"Exported trained learner to {model_path}")

    return learn, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GTSRB classifier.")
    parser.add_argument("--data-id", type=int, default=None, help="Pickle id 0..8.")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50"],
        default=None,
    )
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find before each training phase and print suggestions.",
    )
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.data_id is not None:
        cfg.data_id = args.data_id
    if args.backbone is not None:
        cfg.backbone = args.backbone

    train(cfg, run_lr_find=args.lr_find)


if __name__ == "__main__":
    main()
