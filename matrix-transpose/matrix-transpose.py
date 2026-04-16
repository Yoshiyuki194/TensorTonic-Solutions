import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code 
    n = len(A)
    m = len(A[0])
    return np.array([
        [A[i][j] for i in range(n)]
        for j in range(m)
    ])
