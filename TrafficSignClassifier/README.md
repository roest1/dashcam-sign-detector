# Traffic Sign Classification using Deep Learning

High-accuracy traffic sign classification pipeline using transfer learning with ResNet architectures and large-scale image preprocessing.
Originally developed as a CSC-4740 Big Data project at LSU.

**Contributors:** Riley Oest, Alex Brodsky

---

## 🚦 Project Overview

This project trains deep convolutional neural networks to classify traffic signs across **43 distinct classes** using large preprocessed image datasets (4GB–22GB). The system:

- Converts large pickle datasets into structured image datasets
- Applies preprocessing and selective augmentation
- Fine-tunes pretrained ResNet models using FastAI
- Evaluates performance with accuracy, confusion matrix, ROC/AUC, and classification reports
- Visualizes predictions and failure modes

The final trained model achieves:

> **~97% validation accuracy and ~97% test accuracy** on the 4GB dataset.

---

## 📊 Dataset

Source datasets (Kaggle):

- **22 GB:** 1M+ traffic sign images
- **8 GB:** Light preprocessed dataset
- **4 GB:** Fully preprocessed dataset (used for training)

Each dataset contains:

- Training set
- Validation set
- Test set
- 43 traffic sign labels
- RGB and grayscale variants

Preprocessing variants include:

- Normalization
- Mean/STD normalization
- Histogram equalization
- Grayscale conversion

---

## 🏗️ Architecture

```
Pickle Dataset
     ↓
Image Extraction & Label Structuring
     ↓
FastAI DataBlock Pipeline
     ↓
Pretrained ResNet (34 / 50)
     ↓
Selective Augmentation
     ↓
Fine-Tuning
     ↓
Evaluation + Visualization
```

### Model

- **Backbone:** ResNet-34 / ResNet-50 (ImageNet pretrained)
- **Framework:** FastAI + PyTorch
- **Loss:** CrossEntropy
- **Metrics:** Accuracy, Confusion Matrix, ROC/AUC, Classification Report

---

## 🔬 Training Strategy

### Phase 1 — Frozen Training

- Train only classifier layers
- Identify optimal learning rate
- Initial convergence on unaugmented data

### Phase 2 — Augmented Training

- Introduce selective data augmentation
- Avoid rotation on visually ambiguous signs
- Improve generalization

### Phase 3 — Full Fine-Tuning

- Unfreeze network
- Differential learning rates
- Final convergence and model export

---

## 📈 Results

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | ~97.1% |
| Test Accuracy       | ~97.0% |
| Validation Loss     | ~0.156 |
| Test Loss           | ~0.147 |

### Observations

Some confusion occurs between visually similar signs:

- _Keep Left_ vs _Keep Right_
- _Go Straight or Left_ vs _Go Straight or Right_

This was mitigated by selective rotation augmentation.

---

## 🖼️ Sample Predictions

The project includes tooling to:

- Visualize correct predictions
- Inspect misclassified samples
- Plot confusion matrices
- Generate ROC curves

Example usage is demonstrated in `main.py`.

---

## ⚙️ Repository Structure

```
TrafficSignClassifier/
├── Data/                 # Dataset storage (ignored in git)
├── EMR/                  # Experimental distributed processing scripts
├── main.py               # Training + evaluation pipeline
├── processing.py         # Dataset visualization utilities
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python main.py
```

Datasets must be downloaded separately due to size.

---

## ☁️ Distributed Experimentation (Exploratory)

Early experiments attempted distributed preprocessing using AWS EMR and Spark.
While SSH connectivity was established between nodes, full distributed training was not achieved within project constraints.

Scripts remain in `/EMR` for reference.

---

## 📚 References

Included in `/docs`:

- ResNet architecture overview
- State-of-the-art image classification papers
- YOLO traffic sign research
- Project proposal documentation

---

## 👤 Authors

- **Riley Oest** — Model architecture, preprocessing pipeline, training, evaluation
- **Alex Brodsky** — Distributed experimentation (EMR, SSH clustering)

---

