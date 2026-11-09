import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, l2_penalty=0.0):
        self.lr = learning_rate
        self.l2 = l2_penalty
        self.W = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        # Bound z into [-500, 500] to completely eliminate runtime exp overflow
        z_bounded = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_bounded))

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Prevent log(0) which introduces NaN into training runs
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        
        # Base Binary Cross-Entropy loss
        bce = - (1.0 / m) * np.sum(
            y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe)
        )
        
        # L2 regularizer penalty (ignoring bias term)
        l2_cost = (self.l2 / (2.0 * m)) * np.sum(np.square(self.W)) if self.W is not None else 0.0
        return bce + l2_cost

    def fit(self, X, y, epochs=1000, batch_size=16, shuffle=False):
        m, n = X.shape
        # Initialize zero weights (n features x 1) and zero bias
        self.W = np.zeros((n, 1), dtype=np.float64)
        self.b = 0.0

        for epoch in range(epochs):
            indices = np.arange(m)
            if shuffle:
                np.random.shuffle(indices)
                X_curr = X[indices]
                y_curr = y[indices]
            else:
                X_curr = X
                y_curr = y

            # Mini-batch gradient descent loop
            for start_idx in range(0, m, batch_size):
                end_idx = min(start_idx + batch_size, m)
                X_batch = X_curr[start_idx:end_idx]
                y_batch = y_curr[start_idx:end_idx]
                batch_m = X_batch.shape[0]

                # Forward propagation
                z = np.dot(X_batch, self.W) + self.b
                y_hat = self._sigmoid(z)

                # Gradients: includes standard error term plus L2 weight decay
                error = y_hat - y_batch
                grad_W = (1.0 / batch_m) * np.dot(X_batch.T, error) + (self.l2 / batch_m) * self.W
                grad_b = (1.0 / batch_m) * np.sum(error)

                # Check for numerical explosion/divergence
                if np.isnan(grad_W).any() or np.isinf(grad_W).any():
                    return "diverged"

                # Gradient descent parameter update
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

        return "converged"

    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return self._sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y) * 100.0
