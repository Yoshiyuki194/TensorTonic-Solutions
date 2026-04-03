import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(seq_len).reshape(seq_len, 1)
    num_sin = (d_model + 1) // 2
    indices = np.arange(num_sin).reshape(1, num_sin)
    exponents = 2 * indices / d_model
    divisors = base ** exponents
    angles = positions / divisors
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = sin_angles
    pe[:, 1::2] = cos_angles[:, 0:(d_model // 2)]
    return pe
    