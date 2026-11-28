
import numpy as np

def qb_route(scores, beta, k):
    T, N = scores.shape

    if (T * k) % N != 0:
        raise ValueError("T * k must be divisible by N")

    adjusted = scores - beta

    # Select top k experts for every token
    idx = np.argsort(-adjusted, axis=1, kind="stable")[:, :k]

    # Find the (k+1)-th largest score for every token
    a = np.sort(adjusted, axis=1)[:, -(k + 1)]

    # Calculate margins
    margins = scores - a[:, None]

    # Target capacity per expert
    C = (T * k) // N

    # Find the (C+1)-th largest margin for each expert
    beta_local = np.sort(margins, axis=0)[-(C + 1)]

    return idx, beta_local


def qb_train(steps, k, ema=0.0):
    N = steps[0][0].shape[1]
    beta = np.zeros(N)

    global_maxvio = []
    micro_maxvio = []

    for batch in steps:
        counts = np.zeros(N)
        local_betas = []
        micro_vio = []

        for scores in batch:
            idx, beta_local = qb_route(scores, beta, k)

            # Count expert assignments
            c = np.bincount(idx.ravel(), minlength=N)

            counts += c
            local_betas.append(beta_local)

            # MaxVio for this micro-batch
            avg = np.mean(c)
            micro_vio.append(np.max(c / avg - 1))

        # Update beta using average local bias
        beta_mean = np.mean(local_betas, axis=0)

        beta = ema * beta + (1 - ema) * beta_mean

        # Re-center beta so its sum is zero
        beta -= np.mean(beta)

        # Global batch MaxVio
        avg_global = np.mean(counts)
        global_maxvio.append(round(float(np.max(counts / avg_global - 1)), 4))

        micro_maxvio.append(round(float(np.mean(micro_vio)), 4))

    return {
        "global_maxvio": global_maxvio,
        "micro_maxvio": micro_maxvio,
        "beta": np.round(beta, 4).tolist()
    }
