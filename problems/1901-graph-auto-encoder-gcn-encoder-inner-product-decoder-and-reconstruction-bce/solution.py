import numpy as np


def gae_encode(A, X, W1, W2):
    """
    Two-layer GCN encoder.
    """
    N = A.shape[0]

    # Add self-loops
    A_hat = A + np.eye(N)

    # Calculate degree matrix
    degree = np.sum(A_hat, axis=1)

    # Calculate inverse square root of degrees
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    # Normalize adjacency matrix
    A_hat = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # First GCN layer
    H = A_hat @ X @ W1

    # ReLU activation
    H = np.maximum(0, H)

    # Second GCN layer
    Z = A_hat @ H @ W2

    return Z


def gae_logits(Z, pairs):
    """
    Calculate inner-product logits for node pairs.
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)

    i = pairs[:, 0]
    j = pairs[:, 1]

    return np.sum(Z[i] * Z[j], axis=1)


def gae_loss(A, Z, pos_weight=None):
    """
    Full reconstruction BCE loss using logits.
    """
    N = A.shape[0]

    # Calculate reconstruction logits
    logits = Z @ Z.T

    # Calculate positive and negative counts
    positives = np.sum(A)
    negatives = N * N - positives

    # Calculate positive weight
    if pos_weight is None:
        pos_weight = negatives / positives if positives > 0 else 1.0

    # Numerically stable BCE with logits
    loss = (
        (1 - A) * np.logaddexp(0, logits)
        + pos_weight * A * np.logaddexp(0, -logits)
    )

    return float(np.mean(loss))


def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    """
    Mean BCE loss over sampled positive and negative pairs.
    """
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)

    # Positive samples: -log(sigmoid(logit))
    pos_loss = np.logaddexp(0, -pos_logits)

    # Negative samples: -log(1 - sigmoid(logit))
    neg_loss = np.logaddexp(0, neg_logits)

    # Mean over all samples together
    total_loss = np.concatenate([pos_loss, neg_loss])

    if len(total_loss) == 0:
        raise ValueError("At least one positive or negative pair is required.")

    return float(np.mean(total_loss))


def vgae_sample(mu, logvar, eps):
    """
    Reparameterization trick for VGAE.
    """
    return mu + np.exp(0.5 * logvar) * eps


def vgae_kl(mu, logvar):
    """
    KL divergence between learned Gaussian and standard normal.
    """
    variance = np.exp(logvar)

    kl = -0.5 * np.sum(
        1 + logvar - mu**2 - variance,
        axis=1
    )

    return float(np.mean(kl))