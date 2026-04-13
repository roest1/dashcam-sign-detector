# classifier

FastAI + PyTorch ResNet classifier fine-tuned on the GTSRB 4GB preprocessed
bundle. Inherits the three-phase transfer-learning recipe from the original
TSC class project, which reached ~97% test accuracy on `data0` (shuffled RGB).

## Modules

| file | role |
|---|---|
| `config.py` | `ClassifierConfig` — env-driven paths + hyperparameters |
| `preprocessing.py` | explode GTSRB pickles into `label/image_*.png` trees |
| `dataset.py` | FastAI `DataBlock` builders + `SelectiveRotation` augment |
| `model.py` | backbone resolution (ResNet 18/34/50) + `vision_learner` factory |
| `train.py` | three-phase `fit_one_cycle` schedule |
| `evaluate.py` | accuracy, classification report, confusion matrix, ROC/AUC |

## End-to-end

```bash
# 1. install deps
uv sync

# 2. download the 4GB bundle (see ../../data/README.md for Kaggle token setup)
uv run python scripts/download_data.py

# 3. explode the pickle into PNGs under data/processed/data0images/
uv run python -m dashcam_sign_detector.classifier.preprocessing --data-id 0

# 4. train (3 phases)
uv run python -m dashcam_sign_detector.classifier.train \
    --data-id 0 --backbone resnet34

# 5. evaluate — drops artifacts under reports/data0_resnet34/
uv run python -m dashcam_sign_detector.classifier.evaluate \
    --data-id 0 --backbone resnet34
```

## Configuration

Everything in `ClassifierConfig` can be overridden via env vars:

| env var | default | effect |
|---|---|---|
| `DSD_DATA_ROOT` | `<repo>/data` | root under which `raw/` and `processed/` live |
| `DSD_MODELS_ROOT` | `<repo>/models` | where `classifier_*.pkl` is saved |
| `DSD_REPORTS_ROOT` | `<repo>/reports` | evaluation artifacts |
| `DSD_DATA_ID` | `0` | pickle variant (0..8) |
| `DSD_BACKBONE` | `resnet34` | `resnet18` / `resnet34` / `resnet50` |
| `DSD_IMAGE_SIZE` | `224` | item transform size |
| `DSD_BATCH_SIZE` | `64` | train/valid batch size |
| `DSD_NUM_WORKERS` | `4` | DataLoader workers |

CLI flags on `train.py` / `evaluate.py` override the matching env vars.

## Training schedule

The schedule is load-bearing for reproducing ~97% — don't rewrite to pure
PyTorch in v1.

1. **Phase 1a** — frozen head, unaugmented data, 3 epochs at `lr=2.09e-3`
2. **Phase 1b** — frozen head, selective-rotation aug, 2 epochs at same LR
3. **Phase 2** — unfreeze, differential LR `slice(10^-6.5, 1.32e-6)`, 1 epoch

`SelectiveRotation` skips directional signs (classes 36, 37, 38, 39) so
*Keep-left / Keep-right* and *Go-straight-or-left / Go-straight-or-right*
don't collapse into each other under augmentation.

Re-run `lr_find` with `--lr-find` if you swap the backbone or dataset
variant — the baked-in LRs were tuned for ResNet-34 on `data0`.
