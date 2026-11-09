# Binary Logistic Regression from Scratch

A minimal, pure-NumPy implementation of a binary logistic regression classifier trained via mini-batch gradient descent. Designed for empirical benchmarking on clinical diagnostic data (Pima Indians Diabetes dataset).

---

## Overview

This repository implements binary logistic regression without relying on high-level machine learning libraries (such as `scikit-learn`). All forward passes, gradient calculations, loss evaluations, and optimization loops are vectorized using NumPy.

The implementation supports:

* **Optimization:** Mini-batch gradient descent.


* **Regularization:** $L_2$ weight decay (Ridge penalty).


* **Preprocessing:** Min-max feature normalization fitted strictly on training distributions to prevent data leakage.


* **Numerical Stability:** Exponential bounds on logits and probability clipping against extreme log values.

---

## Mathematical Formulation

### 1. Hypothesis Function

Given an input sample $x \in \mathbb{R}^{d}$, the linear combination and the activation are computed as:

$$z = x^T W + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $W \in \mathbb{R}^{d \times 1}$ is the weight vector, $b \in \mathbb{R}$ is the bias scalar, and $\sigma(z) \in (0, 1)$ represents the posterior probability $P(y = 1 \mid x)$.

### 2. Objective Function (Binary Cross-Entropy with $L_2$ Penalty)

For a batch of $m$ observations, the penalized empirical risk is defined as:

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \Vert{}W\Vert{}_2^2$$

Where:

* $y^{(i)} \in \{0, 1\}$ denotes the binary ground truth.


* $\lambda \ge 0$ is the regularization coefficient.


* The bias term $b$ is unregularized.

### 3. Gradient Updates

The partial derivatives with respect to parameters across a mini-batch of size $m$ are:

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T (\hat{Y} - Y) + \frac{\lambda}{m} W$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

Parameters update iteratively with learning rate $\eta$:

$$W \leftarrow W - \eta \frac{\partial J}{\partial W}$$

$$b \leftarrow b - \eta \frac{\partial J}{\partial b}$$

---

## Repository Structure

```text
.
├── data/
│   ├── train.csv           # 468 training records[cite: 1]
│   ├── validation.csv      # 100 validation records[cite: 2]
│   └── test.csv            # 200 evaluation records[cite: 2]
├── src/
│   ├── __init__.py         # Package exports
│   ├── data_loader.py      # I/O, array formatting, and min-max scaling[cite: 2, 4]
│   ├── metrics.py          # Loss functions and classification accuracy[cite: 2]
│   └── model.py            # LogisticRegression classifier class[cite: 2]
├── run_experiments.py      # Experiment driver covering assignment benchmarks (Q1-Q8)[cite: 3, 4]
├── requirements.txt        # Minimal environment dependencies[cite: 1]
└── README.md

```

---

## Setup

### Requirements

* Python 3.9+
* NumPy $\ge$ 1.20


* Pandas $\ge$ 1.3


* Matplotlib $\ge$ 3.4



```bash
git clone https://github.com/your-username/pima-logistic-scratch.git
cd pima-logistic-scratch
pip install -r requirements.txt

```

---

## Usage Example

```python
from src.data_loader import load_dataset, split_features_target, min_max_normalize
from src.model import LogisticRegression
from src.metrics import compute_accuracy

# 1. Load and split raw data[cite: 2]
train_df = load_dataset("data/train.csv")
val_df = load_dataset("data/validation.csv")

X_train, y_train = split_features_target(train_df)
X_val, y_val = split_features_target(val_df)

# 2. Scale features using training boundaries only[cite: 4]
X_train_norm, X_val_norm, _ = min_max_normalize(X_train, X_val=X_val)

# 3. Instantiate model with L2 regularization[cite: 4]
clf = LogisticRegression(learning_rate=0.01, l2_penalty=0.1)

# 4. Train via mini-batch gradient descent[cite: 2]
status = clf.fit(X_train_norm, y_train, epochs=1000, batch_size=16)

# 5. Evaluate accuracy on unseen validation data[cite: 2]
val_preds = clf.predict(X_val_norm)
accuracy = compute_accuracy(y_val, val_preds)
print(f"Validation Accuracy: {accuracy:.2f}%")

```

---

## Experiments Tracked

The evaluation script `run_experiments.py` reproduces the following benchmarks:

* **Learning Rate Grid Search (Q1):** Tests $\eta \in [10^4, \dots, 10^{-5}]$ across $N \in [10, 100, 1000, 10000]$ iterations to identify divergence boundaries and stable convergence regimes.


* **Learning Curves (Q2 & Q3):** Iteration-by-iteration trajectory plots of empirical risk and accuracy across $\eta \in \{0.1, 0.001, 0.00001\}$.


* **Batch Size Dynamics (Q5):** Compares variance and convergence efficiency for $B \in \{4, 8, 16, 32, 64\}$.


* **$L_2$ Regularization (Q6):** Evaluates generalization delta across penalty terms $\lambda \in [0.001, 10.0]$.


* **Feature Normalization Impact (Q7):** Convergence speed and condition number effects comparing raw versus min-max normalized features.


* **Stochasticity via Shuffling (Q8):** Quantifies variance in final validation metrics over multiple random dataset permutations under mini-batch sampling.
