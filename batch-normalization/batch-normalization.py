import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x, γ, β = np.array(x), np.array(gamma), np.array(beta)
    if x.ndim == 2:
        μ = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        x_hat = (x - μ) / np.sqrt(var + eps)
        return γ * x_hat + β
    elif x.ndim == 4:
        axes = (0, 2, 3)
        μ = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        x_hat = (x - μ) / np.sqrt(var + eps)
        γ = γ.reshape(1, -1, 1, 1)
        β = β.reshape(1, -1, 1, 1)
        return γ * x_hat + β
    else:
        return None