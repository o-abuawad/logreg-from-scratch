import numpy as np
import pandas as pd

def load_dataset(file_path):
    # Read CSV file into a pandas DataFrame[cite: 2]
    df = pd.read_csv(file_path)
    return df

def split_features_target(df, target_column='Outcome'):
    # Separate input features X from binary ground truth label y[cite: 2]
    X_df = df.drop(columns=[target_column])
    y_series = df[target_column]

    # Convert to pure NumPy arrays and shape y as an explicit column vector (m, 1)[cite: 2]
    X = X_df.to_numpy(dtype=np.float64)
    y = y_series.to_numpy(dtype=np.float64).reshape(-1, 1)

    return X, y

def min_max_normalize(X_train, X_val=None, X_test=None):
    # Calculate min and max bounds along each feature strictly on training set[cite: 4]
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)

    # Prevent division by zero if a feature column is constant
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0.0] = 1.0

    # Scale training set to [0, 1][cite: 4]
    X_train_norm = (X_train - min_vals) / range_vals

    # Transform validation and test sets using training parameters (no data leakage)[cite: 4]
    X_val_norm = None
    if X_val is not None:
        X_val_norm = (X_val - min_vals) / range_vals

    X_test_norm = None
    if X_test is not None:
        X_test_norm = (X_test - min_vals) / range_vals

    return X_train_norm, X_val_norm, X_test_norm
