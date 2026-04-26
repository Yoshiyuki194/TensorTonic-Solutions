import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]

    D = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

    same_cluster = labels[:, np.newaxis] == labels[np.newaxis, :]
    intra_mask = same_cluster.copy()
    np.fill_diagonal(intra_mask, False)
    intra_counts = intra_mask.sum(axis=1)
    intra_sum = (D * intra_mask).sum(axis=1)
    a = np.where(intra_counts > 0, intra_sum / intra_counts, 0.0)

    unique_labels = np.unique(labels)
    K = unique_labels.shape[0]
    cluster_masks = (unique_labels[:, np.newaxis] == labels[np.newaxis, :])
    cluster_sums = cluster_masks @ D.T
    cluster_counts = cluster_masks.sum(axis=1, keepdims=True)
    cluster_means = cluster_sums / np.maximum(cluster_counts, 1)
    label_indices = np.searchsorted(unique_labels, labels)
    cluster_means[label_indices, np.arange(n_samples)] = np.inf
    b = cluster_means.min(axis=0)

    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)

    return float(s.mean())
    
    