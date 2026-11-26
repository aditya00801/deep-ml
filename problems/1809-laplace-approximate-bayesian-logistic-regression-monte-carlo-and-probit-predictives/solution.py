import numpy as np


def laplace_logreg(X, y, prior_var, n_iter=50):

    # Dimensions
    n, D = X.shape

    # Initialize weights
    w = np.zeros(D)

    for _ in range(n_iter):

        # Predict probabilities
        mu = 1 / (1 + np.exp(-(X @ w)))

        # Gradient
        gradient = X.T @ (mu - y) + w / prior_var

        # Hessian
        S = mu * (1 - mu)
        H = X.T @ (S[:, None] * X) + np.eye(D) / prior_var

        # Newton update
        step = np.linalg.solve(H, gradient)
        w = w - step

        # Early stopping
        if np.linalg.norm(step) < 1e-12:
            break

    # Final Hessian using optimized weights
    mu = 1 / (1 + np.exp(-(X @ w)))
    S = mu * (1 - mu)

    H = X.T @ (S[:, None] * X) + np.eye(D) / prior_var

    # Covariance
    Sigma = np.linalg.inv(H)

    w_map = w

    return w_map, Sigma
def predictive(X_test, w_map, Sigma, method, n_samples=1000, seed=0):

    if method == "plugin":
        return 1 / (1 + np.exp(-(X_test @ w_map)))

    elif method == "mc":
        rng = np.random.default_rng(seed)

        D = len(w_map)
        Z = rng.standard_normal((n_samples, D))

        L = np.linalg.cholesky(Sigma)
        W = w_map + Z @ L.T

        probabilities = 1 / (1 + np.exp(-(W @ X_test.T)))

        return probabilities.mean(axis=0)

    elif method == "probit":
        m = X_test @ w_map
        v = np.sum((X_test @ Sigma) * X_test, axis=1)

        kappa = 1 / np.sqrt(1 + np.pi * v / 8)

        return 1 / (1 + np.exp(-(kappa * m)))

    else:
        raise ValueError("Unknown method")
