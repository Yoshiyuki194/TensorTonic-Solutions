def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    probs = np.array(prob_distributions)
    tokens = np.array(actual_tokens)
    return np.exp(-np.mean(np.log(probs[np.arange(len(tokens)), tokens])))