# Protein Structure Prediction Pipeline

The dataset comes from the ProteinNet project by Mohammed AlQuraishi:
🔗 https://github.com/aqlaboratory/proteinnet

### CASP9 Dataset · Multi-Model ML/DL Evaluation · GPU-Accelerated

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red?style=flat-square&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

An end-to-end pipeline for **residue-level protein structure prediction** built on the CASP9 dataset. The project covers raw file parsing, feature engineering, distance-based target construction, and systematic evaluation across classical ML, CNN, and transformer-based models.

**Best result:** XGBoost with 861-feature sliding-window representation — **RMSE 577.07 · MAE 160.65 · R² 0.667**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [Setup](#setup)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Challenges & Decisions](#challenges--decisions)

---

## Overview

This project explores whether protein backbone geometry can be predicted from sequence and evolutionary information alone, without access to experimental structure. Each residue in a protein is represented by a feature window derived from its sequence neighbourhood and evolutionary profile. The model then predicts four distance-based structural targets — distances from residue *i* to residues at offsets +1, +2, +4, and +8 — effectively capturing local backbone geometry across multiple scales.

The work progressed through four model families in order of complexity:

1. **Classical baseline** — Linear Regression
2. **Gradient boosting** — XGBoost with GPU acceleration
3. **Deep learning** — 1D CNN architectures
4. **Pretrained sequence models** — ESM / ProtBERT (experimental)

---

## Project Structure

```
MAINFILES/
│
├── casp9/casp9/              # Raw CASP9 training, validation, and test files
│
├── data/
│   ├── Multi_distance_edits/ # Distance target variants and experiments
│   ├── parsed/               # Residue-level Parquet files after initial parsing
│   ├── processed/            # Cleaned and feature-engineered Parquet files
│   └── Saved_Windows_features/ # Precomputed sliding-window feature matrices
│
├── parsingCasp9Dataset/      # CASP9 file parser and tabular conversion scripts
├── Preprocessing/            # Coordinate cleaning, masking, validity filtering
├── FeatureEngineering/       # Sliding-window feature construction pipeline
│
├── ML_Codes/                 # Linear Regression, XGBoost training and evaluation
├── DL_Codes/                 # CNN architectures, MLP, training loops
│
├── ModelOutputs/             # Saved model checkpoints and prediction files
├── Plots/                    # Actual vs predicted curves, residual plots
│
├── GPU_VENV/                 # Virtual environment (not tracked)
├── Gpu_test.py               # GPU availability and CUDA sanity check
└── requirements.txt          # All dependencies
```

---

## Dataset

**Source:** [CASP9 / ProteinNet](https://github.com/aqlaboratory/proteinnet) format

The raw files contain protein entries with the following blocks per protein:

| Block | Content |
|---|---|
| `[ID]` | Protein identifier |
| `[PRIMARY]` | Amino acid sequence (single-letter codes) |
| `[EVOLUTIONARY]` | PSSM-style evolutionary profile — shape `(21, L)` |
| `[TERTIARY]` | 3D backbone coordinates — shape `(L, 9)` |
| `[MASK]` | Residue validity flags |

The dataset uses the **official CASP9 train / validation / test split** — no manual partitioning was applied.

| Split | Proteins | Avg Length | Avg Valid Residues | Valid Ratio |
|---|---|---|---|---|
| Train | 16,973 | 208.5 | 185.8 | 86.1% |
| Validation | 224 | 228.8 | 200.6 | 86.7% |
| Test | 116 | 226.3 | 199.3 | 86.2% |

---

## Pipeline

### 1. Parsing

Raw CASP9 files are parsed into residue-level rows and saved as Parquet:

```
ProteinID | ResidueIndex | Residue | Mask | Evo1...Evo21 | X | Y | Z
```

All `training_*` files are merged into a single unified dataset to maximise training coverage.

### 2. Preprocessing

- Rows with invalid coordinates are removed
- Residue validity is computed as `(Mask == 1) AND (coordinates != all-zero)`
- Proteins with fewer than 10 valid residues are filtered out

### 3. Feature Engineering

A sliding window of configurable size `W` is applied around each residue. For each position in the window, the following are included:

- One-hot encoded residue identity (20 dims)
- Evolutionary profile (21 dims)
- Relative position encoding

Window size 21 (10 residues on each side) with 861 total features per residue gave the best results.

### 4. Target Construction

Four regression targets per residue, derived from 3D coordinates:

```
dist_i1 = ||coords[i] - coords[i+1]||
dist_i2 = ||coords[i] - coords[i+2]||
dist_i4 = ||coords[i] - coords[i+4]||
dist_i8 = ||coords[i] - coords[i+8]||
```

Targets are only computed where both residues are valid. Invalid positions are masked out of the loss.

---

## Models & Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (flat features) | 914.22 | 630.83 | 0.025 |
| XGBoost — early config | 855.11 | 577.79 | 0.147 |
| CNN (initial) | 759.07 | 242.08 | 0.425 |
| CNN (improved) | 757.45 | 204.03 | 0.429 |
| XGBoost — W=11, 451 features | 695.43 | 228.09 | 0.549 |
| XGBoost — W=11, optimised | 629.39 | 175.76 | 0.628 |
| **XGBoost — W=21, 861 features** | **577.07** | **160.65** | **0.667** |

### XGBoost Configuration (best)

```python
XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda",
)
```

### CNN Architecture

```
Input (batch, features, window)
  → Conv1d(64, kernel=3) + ReLU + BatchNorm
  → Conv1d(128, kernel=3) + ReLU + BatchNorm
  → GlobalAvgPool
  → Linear(256) + ReLU + Dropout(0.3)
  → Linear(4)   # dist_i1, dist_i2, dist_i4, dist_i8
```

### Transformer (Experimental)

Attempted frozen **ESM** encoder with a regression head. Encountered NaN loss due to hidden-state instability in the Hugging Face workflow at this data scale. **ProtBERT** (`Rostlab/prot_bert`) is the recommended next step — space-separated residue tokenisation, frozen encoder, same 4-target regression head.

---

## Setup

```bash
git clone https://github.com/your-username/protein-structure-prediction
cd protein-structure-prediction

python -m venv GPU_VENV
source GPU_VENV/bin/activate        # Windows: GPU_VENV\Scripts\activate

pip install -r requirements.txt
```

**Core dependencies:**

```
torch>=2.0          # GPU-enabled build recommended
xgboost>=2.0
pandas
numpy
pyarrow
scikit-learn
transformers        # for ProtBERT
```

Verify GPU availability:

```bash
python Gpu_test.py
```

---

## Usage

### Parse raw CASP9 files

```bash
cd parsingCasp9Dataset
python parse_all_training.py      # produces data/parsed/*.parquet
```

### Preprocess and engineer features

```bash
cd Preprocessing
python clean_coords.py            # removes invalid residues

cd FeatureEngineering
python build_features.py --window 21   # produces data/processed/*.parquet
```

### Train XGBoost

```bash
cd ML_Codes
python train_xgboost.py --window 21 --features 861
```

### Train CNN

```bash
cd DL_Codes
python train_cnn.py --epochs 50 --batch_size 2048
```

### Evaluate on test set

```bash
python evaluate.py --model xgboost --split test
```

---

## Key Findings

**Feature engineering mattered more than model architecture.** Moving from a flat feature representation to a window-21 sliding-window setup reduced RMSE from 855 to 577 — a larger gain than any model change alone.

**XGBoost outperformed deep learning** in this constrained setup. With 8 GB VRAM and CPU-based preprocessing, gradient boosting on hand-engineered features was both faster and more accurate than CNN or transformer approaches.

**Local context is structurally meaningful.** Each increase in window size from 5 to 21 produced consistent RMSE reductions, suggesting that roughly 10 residues on each side carry useful geometric information.

**Transformer models are theoretically superior but practically demanding.** ESM produced NaN outputs under the current workflow. ProtBERT is the recommended retry path with careful tokenisation, padding strategy, and frozen-encoder training.

---

## Challenges & Decisions

| Challenge | Resolution |
|---|---|
| Single-class label — classification infeasible | Pivoted to regression with distance-based targets |
| Random Forest crashed (OOM) | Switched to XGBoost with GPU |
| `gpu_hist` deprecated in XGBoost ≥ 2.0 | Updated to `tree_method=hist, device=cuda` |
| CNN channel-dimension mismatch | Corrected tensor layout to `(batch, features, window)` |
| ESM NaN predictions despite debugging | Pivoted to ProtBERT recommendation |
| SHAP errors with multi-output XGBoost | Dropped SHAP; focused on quantitative metrics |
| RAPIDS/cuML required CUDA 12.1 | Stayed with pandas/numpy + XGBoost GPU |

---

## License

MIT — free to use, modify, and build on.

---

*Built as part of a structured ML research project on protein structure prediction using the CASP9 benchmark dataset.*
