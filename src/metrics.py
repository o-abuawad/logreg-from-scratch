import numpy as np

def compute_loss(y_true, y_pred, W=None, l2_penalty=0.0):
    m = y_true.shape[0]
    if m == 0:
        return 0.0

    # Clip probabilities to avoid log(0) causing NaN or -inf errors
    eps = 1e-15
    y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)

    # Binary cross-entropy cost function
    bce = - (1.0 / m) * np.sum(
        y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe)
    )

    # Optional L2 regularization weight penalty term[cite: 4]
    if W is not None and l2_penalty > 0.0:
        l2_cost = (l2_penalty / (2.0 * m)) * np.sum(np.square(W))
        return float(bce + l2_cost)

    return float(bce)

def compute_accuracy(y_true, y_pred):
    if y_true.shape[0] == 0:
        return 0.0

    # Ensure integer labels for binary matching[cite: 2]
    y_pred_binary = (y_pred >= 0.5).astype(int)
    y_true_binary = y_true.astype(int)

    # Percentage of correctly matched instances[cite: 2]
    return float(np.mean(y_pred_binary == y_true_binary) * 100.0)
