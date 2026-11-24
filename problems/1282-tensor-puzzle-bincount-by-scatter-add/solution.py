import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    """Count occurrences of 0..n_bins-1 in integer array a."""
    counts = np.zeros(n_bins,dtype = int )

    np.add.at(counts, a, 1)  # Add 1 to every index specified in a.

    return counts