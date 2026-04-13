"""Model factory for the GTSRB classifier.

Keeps the transfer-learning recipe from the original TSC project:
FastAI ``vision_learner`` over an ImageNet-pretrained ResNet backbone. The
hand-tuned ``fit_one_cycle`` schedule in ``train.py`` is load-bearing for
hitting the ~97% baseline, so the backbone choice lives here to keep model
construction and training coupled but legible.
"""

from __future__ import annotations

from fastai.vision.all import (
    CrossEntropyLossFlat,
    DataLoaders,
    Learner,
    accuracy,
    vision_learner,
)
from torchvision import models as tvm

from dashcam_sign_detector.classifier.config import ClassifierConfig

_BACKBONES = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
}


def resolve_backbone(name: str):
    try:
        return _BACKBONES[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown backbone {name!r}. Known: {sorted(_BACKBONES)}"
        ) from e


def build_learner(cfg: ClassifierConfig, dls: DataLoaders, n_classes: int) -> Learner:
    arch = resolve_backbone(cfg.backbone)
    return vision_learner(
        dls,
        arch,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        n_in=cfg.color_channels,
        n_out=n_classes,
        pretrained=True,
    )
