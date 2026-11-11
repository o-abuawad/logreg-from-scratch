# Logistic Regression Classifier

A scratch implementation of binary logistic regression using mini-batch gradient descent to predict diabetes onset from clinical diagnostic data. Built strictly using NumPy and Pandas without high-level machine learning frameworks.

---

## Overview

The model optimizes binary cross-entropy loss over clinical diagnostic attributes to classify outcomes ($y \in \{0, 1\}$). The pipeline covers data ingestion, gradient descent optimization, hyperparameter tuning, L2 regularization, feature scaling, and sample shuffling.

### Key Components

- **Loss Function:** Binary Cross-Entropy (Log Loss).
- **Optimization:** Mini-Batch Gradient Descent (vectorized).
- **Evaluation Metric:** Classification Accuracy ($\%$) on train, validation (dev), and test splits.
- **Regularization:** $L_2$ weight decay penalty on weight vectors ($w$).
- **Preprocessing:** Min-Max feature normalization computed strictly on training distributions.

---

## Project Structure

```text
.
├── diabetes_train.csv       # Training split (468 records)
├── diabetes_dev.csv         # Validation split (100 records)
├── diabetes_test.csv        # Test split (200 records)
├── main.py                  # Full implementation and experimental harness
├── requirements.txt         # Environment dependencies
└── README.md

```

---

## Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `matplotlib`

Install dependencies:

```bash
pip install numpy pandas matplotlib

```

---

## Usage

Ensure the dataset CSV files (`diabetes_train.csv`, `diabetes_dev.csv`, `diabetes_test.csv`) reside in the root directory alongside the source file.

Run the entire pipeline:

```bash
python3 main.py

```

---

## Pipeline & Experiments

The script executes the following sequential experimental modules:

1. **Learning Rate & Iteration Grid Search (`run_q1`)**
* Evaluates learning rates $\eta \in [10^{-5}, 10^4]$ across $[10, 100, 1000, 10000]$ iterations with batch size $16$.
* Tracks convergence boundaries, divergence thresholds, and numerical overflow points.


2. **Convergence Dynamics (`run_q2_q3_q4`)**
* Plots loss and accuracy trajectories over $1000$ iterations for $\eta \in \{0.1, 0.001, 0.00001\}$.
* Evaluates the optimal unnormalized baseline configuration on the test set.


3. **Batch Size Sensitivity (`run_q5`)**
* Compares validation performance across mini-batch sizes: $B \in \{4, 8, 16, 32, 64\}$ at $\eta = 0.01$.


4. **$L_2$ Regularization (`run_q6`)**
* Implements parameter shrinkage: $\nabla_w \mathcal{L}_{reg} = \nabla_w \mathcal{L} + \frac{\lambda}{m} w$.
* Tests regularization strengths $\lambda \in [0.0, 10.0]$ to evaluate generalization and prevent overfitting.


5. **Min-Max Feature Scaling (`run_q7`)**
* Scales input features to $[0, 1]$ using training set extrema:

$$X_{norm} = \frac{X - X_{min}^{train}}{X_{max}^{train} - X_{min}^{train}}$$


* Prevents vanishing/exploding gradients across heterogeneous measurement scales.


6. **Stochastic Ordering / Shuffling (`run_q8`)**
* Tests sensitivity to mini-batch sample ordering using multiple random seed permutations ($10, 42, 99$).



---

## Output Artifacts

Running the script outputs terminal performance tables along with the following diagnostic plots:

* `Q2_acc_lr_<lr>.png` — Accuracy curves per learning rate.
* `Q3_loss_lr_<lr>.png` — Loss curves per learning rate.
* `Q5_batch_sizes.png` — Validation accuracy trajectories across batch sizes.
* `Q6_regularization.png` — Accuracy vs. regularization strength ($\lambda$).
* `Q7_norm_acc_lr_<lr>.png` — Convergence trajectories with normalized features.
