"""Evaluation metrics for load forecasting."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return MSE, RMSE, MAE, and MAPE."""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    error = predicted - actual
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    nonzero = np.abs(actual) > 1e-12
    if np.any(nonzero):
        mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
    else:
        mape = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
