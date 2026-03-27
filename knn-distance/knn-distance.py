import numpy as np

def knn_distance(X_train, X_test, k):
    """Compute pairwise distances and return k nearest neighbor indices."""
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D arrays by reshaping to 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Compute pairwise squared Euclidean distances with broadcasting
    diff = X_test[:, None, :] - X_train[None, :, :]
    dist = np.sum(diff ** 2, axis=2)

    # Sort neighbors by distance
    sorted_idx = np.argsort(dist, axis=1)

    # Handle k larger than training set size by padding with -1
    k_eff = min(k, n_train)
    result = np.full((n_test, k), -1, dtype=int)
    result[:, :k_eff] = sorted_idx[:, :k_eff]

    return result
