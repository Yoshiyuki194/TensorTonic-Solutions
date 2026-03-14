def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    if step < warmup_steps:
        X = np.array([0.0, warmup_steps])
        Y = np.array([0.0, initial_lr])
        return np.interp(step, X, Y)
    elif step > total_steps:
        return final_lr
    else:
        X = np.array([warmup_steps, total_steps])
        Y = np.array([initial_lr, final_lr])
        return np.interp(step, X, Y)
    