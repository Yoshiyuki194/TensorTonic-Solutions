import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes is None:
        if y_true.size == 0:
            raise ValueError("num_classes must be provided when y_true is empty")
        K = int(max(y_true.max(), y_pred.max()) + 1)
    else:
        K = int(num_classes)
        if K <= 0:
            raise ValueError("num_classes must be positive")

    if y_true.size == 0:
        cm = np.zeros((K, K), dtype=int)
        if normalize == 'none':
            return cm
        return cm.astype(float)

    if y_true.min() < 0 or y_pred.min() < 0:
        raise ValueError("Labels must be non-negative integers")
    if y_true.max() >= K or y_pred.max() >= K:
        raise ValueError("Labels must be in the range [0, num_classes-1]")

    idx = y_true * K + y_pred
    cm = np.bincount(idx, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm

    cm = cm.astype(float)

    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0 
        cm = cm / row_sums

    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        cm = cm / col_sums

    elif normalize == 'all':
        total = cm.sum()
        if total == 0.0:
            total = 1.0
        cm = cm / total

    else:
        raise ValueError("normalize must be one of 'none', 'true', 'pred', 'all'")

    return cm