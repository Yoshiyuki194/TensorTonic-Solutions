import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    w, m, v, grad = np.array(w), np.array(m), np.array(v), np.array(grad)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * np.square(grad)
    w = w - lr * ((beta1 * m) + (1 - beta1) * grad) / (np.sqrt(v) + eps)
    return w, m, v