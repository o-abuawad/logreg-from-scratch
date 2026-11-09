# Team : 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("diabetes_train.csv")
test = pd.read_csv("diabetes_test.csv")
val = pd.read_csv("diabetes_dev.csv")

np.seterr(over="ignore", invalid="ignore", divide="ignore")

train_X = train.drop(columns=["Outcome"]).values
train_Y = train["Outcome"].values.reshape(-1, 1)

test_X = test.drop(columns=["Outcome"]).values
test_Y = test["Outcome"].values.reshape(-1, 1)

val_X = val.drop(columns=["Outcome"]).values
val_Y = val["Outcome"].values.reshape(-1, 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_accuracy(X, Y, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    if np.isnan(probs).any() or np.isinf(probs).any():
        return np.nan
    preds = (probs >= 0.5).astype(int)
    return np.mean(preds == Y) * 100

def compute_loss(X, Y, w, b):
    m = X.shape[0]
    probs = sigmoid(np.dot(X, w) + b)
    loss = -np.sum(Y * np.log(probs) + (1 - Y) * np.log(1 - probs)) / m
    return loss

def train_model(train_X, train_Y, val_X, val_Y, lr=0.001, iterations=10, batch_size=16, print_batches=False):
    n_samples, n_features = train_X.shape
    w = np.zeros((n_features, 1))
    b = 0.0
    diverged = False

    for it in range(1, iterations + 1):
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]
            m_batch = X_batch.shape[0]

            z = np.dot(X_batch, w) + b
            probs = sigmoid(z)

            if np.isnan(probs).any() or np.isinf(probs).any():
                diverged = True
                break

            dz = probs - Y_batch
            dw = np.dot(X_batch.T, dz) / m_batch
            db = np.sum(dz) / m_batch

            w = w - lr * dw
            b = b - lr * db

            if print_batches:
                batch_loss = compute_loss(X_batch, Y_batch, w, b)
                batch_acc = compute_accuracy(X_batch, Y_batch, w, b)
                print(f"Iter {it} | Batch {start // batch_size + 1} | Loss: {batch_loss:.4f} | Train Acc: {batch_acc:.2f}%")

        if diverged or np.isnan(w).any() or np.isinf(w).any():
            diverged = True
            break

    train_acc = np.nan if diverged else compute_accuracy(train_X, train_Y, w, b)
    val_acc = np.nan if diverged else compute_accuracy(val_X, val_Y, w, b)
    return w, b, train_acc, val_acc, diverged

w, b, tr_acc, val_acc, div = train_model(train_X, train_Y, val_X, val_Y, lr=0.0001, iterations=2, batch_size=16, print_batches=True)
print(f"Final Train Acc: {tr_acc:.2f}%, Val Acc: {val_acc:.2f}%")

def run_q1():
    learning_rates = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    iteration_list = [10, 100, 1000, 10000]

    train_table = pd.DataFrame(index=learning_rates, columns=iteration_list)
    val_table = pd.DataFrame(index=learning_rates, columns=iteration_list)
    overflowed = []

    for lr in learning_rates:
        lr_diverged = False
        for it in iteration_list:
            w_q1, b_q1, tr_acc, v_acc, div = train_model(train_X, train_Y, val_X, val_Y, lr=lr, iterations=it, batch_size=16)
            loss_q1 = compute_loss(train_X, train_Y, w_q1, b_q1)
            if div or np.isnan(tr_acc) or np.isnan(loss_q1) or np.isinf(loss_q1):
                train_table.loc[lr, it] = "Diverged"
                val_table.loc[lr, it] = "Diverged"
                lr_diverged = True
            else:
                train_table.loc[lr, it] = f"{tr_acc:.2f}%"
                val_table.loc[lr, it] = f"{v_acc:.2f}%"
        if lr_diverged:
            overflowed.append(lr)

    print("\nTraining Accuracy (%):")
    print(train_table)
    print("\nValidation Accuracy (%):")
    print(val_table)
    print("\nOverflowed / Diverged Learning Rates:", overflowed)

run_q1()

def train_with_history(train_X, train_Y, val_X, val_Y, lr, iterations=1000, batch_size=16):
    n_samples, n_features = train_X.shape
    w = np.zeros((n_features, 1))
    b = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    for it in range(1, iterations + 1):
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]
            m_batch = X_batch.shape[0]

            z = np.dot(X_batch, w) + b
            probs = sigmoid(z)
            dz = probs - Y_batch
            dw = np.dot(X_batch.T, dz) / m_batch
            db = np.sum(dz) / m_batch

            w = w - lr * dw
            b = b - lr * db

        tr_loss = compute_loss(train_X, train_Y, w, b)
        v_loss = compute_loss(val_X, val_Y, w, b)
        tr_acc = compute_accuracy(train_X, train_Y, w, b)
        v_acc = compute_accuracy(val_X, val_Y, w, b)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        train_accs.append(tr_acc)
        val_accs.append(v_acc)

    return train_accs, val_accs, train_losses, val_losses

def run_q2_q3_q4():
    lrs = [0.1, 0.001, 0.00001]
    iters = np.arange(1, 1001)

    for lr in lrs:
        tr_accs, v_accs, tr_losses, v_losses = train_with_history(train_X, train_Y, val_X, val_Y, lr=lr, iterations=1000, batch_size=16)

        plt.figure(figsize=(8, 5))
        plt.plot(iters, tr_accs, label="Training Accuracy")
        plt.plot(iters, v_accs, label="Validation Accuracy")
        plt.title(f"Accuracy vs Iterations (lr = {lr})")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Q2_acc_lr_{lr}.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(iters, tr_losses, label="Average Training Loss")
        plt.plot(iters, v_losses, label="Average Validation Loss")
        plt.title(f"Average Loss vs Iterations (lr = {lr})")
        plt.xlabel("Iterations")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Q3_loss_lr_{lr}.png")
        plt.close()

    w_best, b_best, _, _, _ = train_model(train_X, train_Y, val_X, val_Y, lr=0.001, iterations=10000, batch_size=16)
    test_acc = compute_accuracy(test_X, test_Y, w_best, b_best)
    print(f"\nQ4 Testing Accuracy (lr=0.001, iterations=10000): {test_acc:.2f}%")

run_q2_q3_q4()

def run_q5():
    batch_sizes = [4, 8, 16, 32, 64]
    iters = np.arange(1, 1001)

    plt.figure(figsize=(9, 6))
    for bs in batch_sizes:
        _, v_accs, _, _ = train_with_history(train_X, train_Y, val_X, val_Y, lr=0.01, iterations=1000, batch_size=bs)
        plt.plot(iters, v_accs, label=f"Batch Size = {bs} (Final: {v_accs[-1]:.2f}%)")
        print(f"Q5 Batch Size {bs}: Final Validation Accuracy = {v_accs[-1]:.2f}%")

    plt.title("Validation Accuracy vs Iterations for Different Batch Sizes (lr=0.01)")
    plt.xlabel("Iterations")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q5_batch_sizes.png")
    plt.close()

def train_model_l2(train_X, train_Y, lr=0.01, iterations=1000, batch_size=16, l2_lambda=0.0):
    n_samples, n_features = train_X.shape
    w = np.zeros((n_features, 1))
    b = 0.0

    for it in range(1, iterations + 1):
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]
            m_batch = X_batch.shape[0]

            z = np.dot(X_batch, w) + b
            probs = sigmoid(z)
            dz = probs - Y_batch
            dw = (np.dot(X_batch.T, dz) / m_batch) + (l2_lambda / m_batch) * w
            db = np.sum(dz) / m_batch

            w = w - lr * dw
            b = b - lr * db

    return w, b

def run_q6():
    lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    tr_accs, val_accs, test_accs = [], [], []

    print("\nQ6 L2 Regularization Results (lr=0.01, iterations=1000):")
    for lam in lambdas:
        w, b = train_model_l2(train_X, train_Y, lr=0.01, iterations=1000, batch_size=16, l2_lambda=lam)
        tr_acc = compute_accuracy(train_X, train_Y, w, b)
        v_acc = compute_accuracy(val_X, val_Y, w, b)
        t_acc = compute_accuracy(test_X, test_Y, w, b)
        tr_accs.append(tr_acc)
        val_accs.append(v_acc)
        test_accs.append(t_acc)
        print(f"Lambda = {lam}: Train Acc = {tr_acc:.2f}%, Val Acc = {v_acc:.2f}%, Test Acc = {t_acc:.2f}%")

    plt.figure(figsize=(8, 5))
    x_labels = [str(l) for l in lambdas]
    plt.plot(x_labels, tr_accs, marker="o", label="Training Accuracy")
    plt.plot(x_labels, val_accs, marker="s", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy vs L2 Regularization (Lambda)")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q6_regularization.png")
    plt.close()

run_q5()
run_q6()

def min_max_normalize(train_X, val_X, test_X):
    min_vals = np.min(train_X, axis=0)
    max_vals = np.max(train_X, axis=0)
    range_vals = max_vals - min_vals
    norm_train_X = (train_X - min_vals) / range_vals
    norm_val_X = (val_X - min_vals) / range_vals
    norm_test_X = (test_X - min_vals) / range_vals
    return norm_train_X, norm_val_X, norm_test_X

def run_q7():
    norm_train_X, norm_val_X, norm_test_X = min_max_normalize(train_X, val_X, test_X)
    lrs = [0.1, 0.01, 0.001]
    iters = np.arange(1, 1001)

    print("\nQ7 Min-Max Normalization Results (iterations=1000, batch_size=16):")
    for lr in lrs:
        tr_accs, v_accs, _, _ = train_with_history(norm_train_X, train_Y, norm_val_X, val_Y, lr=lr, iterations=1000, batch_size=16)
        w_norm, b_norm, _, _, _ = train_model(norm_train_X, train_Y, norm_val_X, val_Y, lr=lr, iterations=1000, batch_size=16)
        t_acc = compute_accuracy(norm_test_X, test_Y, w_norm, b_norm)
        print(f"Normalized lr = {lr}: Train Acc = {tr_accs[-1]:.2f}%, Val Acc = {v_accs[-1]:.2f}%, Test Acc = {t_acc:.2f}%")

        plt.figure(figsize=(8, 5))
        plt.plot(iters, tr_accs, label="Training Accuracy (Normalized)")
        plt.plot(iters, v_accs, label="Validation Accuracy (Normalized)")
        plt.title(f"Normalized Data: Accuracy vs Iterations (lr = {lr})")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Q7_norm_acc_lr_{lr}.png")
        plt.close()

def run_q8():
    seeds = [10, 42, 99]
    print("\nQ8 Shuffling Results (lr=0.01, iterations=1000, batch_size=16):")
    for i, seed in enumerate(seeds, 1):
        np.random.seed(seed)
        indices = np.random.permutation(train_X.shape[0])
        shuffled_X = train_X[indices]
        shuffled_Y = train_Y[indices]
        _, _, tr_acc, v_acc, _ = train_model(shuffled_X, shuffled_Y, val_X, val_Y, lr=0.01, iterations=1000, batch_size=16)
        print(f"Shuffle Run {i} (seed={seed}): Train Acc = {tr_acc:.2f}%, Val Acc = {v_acc:.2f}%")

run_q7()
run_q8()