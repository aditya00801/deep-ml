import numpy as np

def manual_cross_entropy_backward(logits, y):
    n, C = logits.shape

    # Forward pass
    logit_maxes = logits.max(axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)

    # Backward pass
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[np.arange(n), y] = -1 / n

    dprobs = dlogprobs / probs

    dcounts_sum_inv = (counts * dprobs).sum(
        axis=1, keepdims=True
    )

    dcounts_sum = -dcounts_sum_inv / (counts_sum ** 2)

    dcounts = (
        counts_sum_inv * dprobs
        + dcounts_sum
    )

    dnorm_logits = counts * dcounts

    dlogit_maxes = -dnorm_logits.sum(
        axis=1, keepdims=True
    )

    dlogits = dnorm_logits.copy()

    max_indices = np.argmax(logits, axis=1)
    np.add.at(
        dlogits,
        (np.arange(n), max_indices),
        dlogit_maxes.ravel()
    )

    return {
        'dlogprobs': dlogprobs,
        'dprobs': dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum': dcounts_sum,
        'dcounts': dcounts,
        'dnorm_logits': dnorm_logits,
        'dlogit_maxes': dlogit_maxes,
        'dlogits': dlogits
    }