# Recyclable and Household Waste Classification with Deep Learning

Deep learning course team project for **COMPSCI 590: Principles of Deep Learning**.

## Overview

This project studies **multiclass image classification** on a recyclable and household waste dataset. Our goal is not only to build an accurate classifier, but also to understand **how architectural choices and hyperparameters affect optimization, generalization, and calibration**.

This project follows the **Type I** project format for the course:

- find a problem and dataset
- design a neural network architecture
- tune hyperparameters
- explain what worked
- explain what did not work
- try to explain why

## Problem Statement

We aim to classify waste images into one of **30 categories** using convolutional neural networks (CNNs). The broader motivation is to support more robust waste sorting and recycling systems.

Beyond final accuracy, we study:

- shallow vs deeper CNN architectures
- batch normalization vs no batch normalization
- optimizer choice: Adam vs SGD / momentum
- augmentation and regularization effects
- calibration and confidence quality

## Dataset

We use the **Recyclable and Household Waste Classification Dataset** from Kaggle:

**Dataset link:**  
<https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification>

### Dataset summary

- **15,000 images**
- **30 classes**
- **500 images per class**
- image size: **256 x 256**
- each class contains:
  - `default/` images
  - `real_world/` images

### Example classes

- aerosol cans
- aluminum food cans
- aluminum soda cans
- cardboard boxes
- cardboard packaging
- clothing
- coffee grounds
- disposable plastic cutlery
- eggshells
- food waste
- glass beverage bottles
- glass cosmetic containers
- glass food jars
- magazines
- newspaper
- office paper
- paper cups
- plastic cup lids
- plastic detergent bottles
- plastic food containers
- plastic shopping bags
- plastic soda bottles
- plastic straws
- plastic trash bags
- plastic water bottles
- shoes
- steel food cans
- styrofoam cups
- styrofoam food containers
- tea bags

### Note on splitting

The dataset does **not** come with a predefined train/validation/test split, so we create the split manually and keep it fixed across all experiments for fair comparison.

## Project Goals

The main goals of this project are:

1. Build a strong baseline CNN for waste classification.
2. Design and compare improved CNN architectures.
3. Study how hyperparameters affect training and validation behavior.
4. Analyze failure cases and confusing class pairs.
5. Evaluate both **accuracy** and **calibration**.

## Research Questions (to be determined)

We organize the project around the following questions:

1. Does a deeper CNN outperform a shallow CNN on this dataset?
2. Does batch normalization improve training stability and validation accuracy?
3. How do Adam and SGD with momentum compare in convergence and generalization?
4. Does data augmentation improve performance on real-world waste images?
5. Can a model with high accuracy still be poorly calibrated?

## Hypotheses (to be determined)

- **H1:** A deeper CNN with batch normalization will outperform a shallow CNN.
- **H2:** Data augmentation will improve generalization, especially on real-world images.
- **H3:** Adam will converge faster, while tuned SGD with momentum may generalize better.
- **H4:** The best accuracy model may still be overconfident, and temperature scaling can improve calibration.

## Planned Experiments

### 1. Baseline model
A simple CNN with a small number of convolutional blocks to establish a reference point.

### 2. Architecture comparison
Compare:
- shallow CNN
- deeper CNN
- deeper CNN + batch normalization

### 3. Optimizer comparison
Compare:
- Adam
- SGD
- SGD with momentum

### 4. Regularization comparison
Compare:
- no dropout vs dropout
- no weight decay vs weight decay
- no augmentation vs light augmentation

### 5. Calibration analysis
For the best-performing models, evaluate:
- confidence distribution
- reliability behavior
- calibration before and after temperature scaling

## Metrics

We plan to report:

- accuracy
- macro F1 score
- per-class accuracy
- confusion matrix
- training / validation loss curves
- training / validation accuracy curves
- calibration plots
- expected calibration error (if implemented)

## Repository Structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── src/
│   ├── datasets.py
│   ├── transforms.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   └── utils.py
├── experiments/
│   ├── baseline/
│   ├── architecture/
│   ├── optimizers/
│   ├── regularization/
│   └── calibration/
├── results/
│   ├── checkpoints/
│   ├── plots/
│   └── tables/
└── report/
```

# Images

Images come from **Recyclable and Household Waste Classification** on Kaggle ([dataset page](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification), by Alistair King).

# `split_dataset.py`

Splits each class’s `default` and `real_world` folders into train/test (no overlap between splits; both domains represented when present). Requires **Python 3** (stdlib only).

**Defaults:** `--source` is `images` (nested `images/images/<class>/...` is detected automatically); `--output` is `data`, creating `data/training data/` and `data/testing data/`; `--train-ratio` is `0.8`; `--seed` is `42`. Remove existing output dirs or use another `--output` if those folders already exist.

```bash
python3 split_dataset.py
python3 split_dataset.py --source images --output data --train-ratio 0.8 --seed 42 -v
python3 split_dataset.py --no-recursive   # images only directly under default/ and real_world/
```

| Option | Description |
|--------|-------------|
| `--source` | Root with one folder per class (default: `images`) |
| `--output` | Root for `training data` / `testing data` (default: `data`) |
| `--train-ratio` | Train fraction in `(0, 1)` |
| `--seed` | Random seed |
| `--no-recursive` | Do not recurse into subfolders of default/real_world |
| `-v` | Print per-class image counts |
