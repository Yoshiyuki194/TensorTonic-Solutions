import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X)
    if X.shape[0] < 2 or X.ndim != 2:
        return None

    μ = np.mean(X, axis=0)
    X_centered = X - μ

    try:
        return (1 / (X.shape[0] - 1)) * np.matmul(np.transpose(X_centered), X_centered)
    except:
        return None