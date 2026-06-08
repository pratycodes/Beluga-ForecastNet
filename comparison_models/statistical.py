"""Statistical forecasting baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatisticalForecast:
    model_name: str
    predictions: np.ndarray
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None = None


def fit_arima_forecast(
    y_train: np.ndarray,
    forecast_steps: int,
    order: tuple[int, int, int] = (2, 1, 2),
) -> StatisticalForecast:
    """Fit ARIMA and forecast future validation steps."""

    logger.info("Fitting ARIMA order=%s train_samples=%s", order, len(y_train))
    model = _sarimax(y_train, order=order, seasonal_order=(0, 0, 0, 0))
    fitted = model.fit(disp=False)
    predictions = np.asarray(fitted.forecast(steps=forecast_steps), dtype=float)
    logger.info("Finished ARIMA forecast steps=%s", forecast_steps)
    return StatisticalForecast(
        model_name="ARIMA",
        predictions=predictions,
        order=order,
    )


def fit_sarima_forecast(
    y_train: np.ndarray,
    forecast_steps: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
) -> StatisticalForecast:
    """Fit SARIMA and forecast future validation steps."""

    logger.info(
        "Fitting SARIMA order=%s seasonal_order=%s train_samples=%s",
        order,
        seasonal_order,
        len(y_train),
    )
    model = _sarimax(y_train, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    predictions = np.asarray(fitted.forecast(steps=forecast_steps), dtype=float)
    logger.info("Finished SARIMA forecast steps=%s", forecast_steps)
    return StatisticalForecast(
        model_name="SARIMA",
        predictions=predictions,
        order=order,
        seasonal_order=seasonal_order,
    )


def _sarimax(
    y_train: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for ARIMA/SARIMA baselines. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    return SARIMAX(
        np.asarray(y_train, dtype=float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
