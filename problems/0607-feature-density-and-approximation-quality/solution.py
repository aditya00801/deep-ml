import numpy as np

def feature_density_approximation(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    test_values: np.ndarray,
    n_centers: int,
    sigma: float
) -> dict:

    # Generate centers
    if n_centers == 1:
        centers = np.array([0.5])
    else:
        centers = np.linspace(0, 1, n_centers)

    # Calculate training features
    Phi = np.exp(
        -((train_states[:, None] - centers) ** 2) / (2 * sigma ** 2)
    )

    # Fit weights
    weights = np.linalg.lstsq(Phi, train_values, rcond=None)[0]

    # Calculate test features
    test_Phi = np.exp(
        -((test_states[:, None] - centers) ** 2) / (2 * sigma ** 2)
    )

    # Predict values
    predictions = test_Phi @ weights

    # Calculate RMSE
    rmse = np.sqrt(np.mean((test_values - predictions) ** 2))

    return {
        "centers": np.round(centers, 4).tolist(),
        "weights": np.round(weights, 4).tolist(),
        "predictions": np.round(predictions, 4).tolist(),
        "rmse": round(float(rmse), 4)
    }
