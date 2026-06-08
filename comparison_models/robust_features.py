"""GEFCom2014 feature preparation for proposed and baseline experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfs_hdlbwo.gefcom_dataset import build_gefcom_load_features

logger = logging.getLogger(__name__)

FEATURE_MODE_GEFCOM_LOAD = "gefcom_load"
FEATURE_MODES = (FEATURE_MODE_GEFCOM_LOAD,)


@dataclass(frozen=True)
class ZScoreScaler:
    mean: pd.Series
    std: pd.Series


@dataclass(frozen=True)
class RobustPreparedDataset:
    features: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_tune: np.ndarray
    y_tune: np.ndarray
    y_train_real: np.ndarray
    y_fit_real: np.ndarray
    y_tune_real: np.ndarray
    y_val_real: np.ndarray
    y_train_sequence: np.ndarray
    y_fit_sequence: np.ndarray
    y_tune_sequence: np.ndarray
    y_val_sequence: np.ndarray
    feature_scaler: ZScoreScaler
    target_mean: float
    target_std: float
    feature_names: tuple[str, ...]
    feature_mode: str
    metadata: dict[str, Any]

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.target_std + self.target_mean


def prepare_robust_dataset(
    dataset_path: str | Path,
    window_size: int = 72,
    forecast_horizon: int = 1,
    validation_split: float = 0.2,
    tune_split: float = 0.2,
    feature_mode: str = FEATURE_MODE_GEFCOM_LOAD,
    start_date: str | None = None,
    end_date: str | None = None,
    recent_years: int | None = 1,
) -> RobustPreparedDataset:
    """Prepare GEFCom windows with train-only z-score scaling."""

    if feature_mode != FEATURE_MODE_GEFCOM_LOAD:
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}; expected 'gefcom_load'")

    features = build_gefcom_load_features(
        dataset_path,
        start_date=start_date,
        end_date=end_date,
        recent_years=recent_years,
    )
    x_raw, y_raw = create_raw_windows(
        features,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_feature="load",
    )
    x_train_raw, y_train_real, x_val_raw, y_val_real = split_windows(
        x_raw,
        y_raw,
        validation_split=validation_split,
    )
    x_fit_raw, y_fit_real, x_tune_raw, y_tune_real = split_windows(
        x_train_raw,
        y_train_real,
        validation_split=tune_split,
    )

    feature_scaler = fit_feature_scaler(x_fit_raw, features.columns)
    target_mean = float(np.mean(y_fit_real))
    target_std = float(np.std(y_fit_real)) or 1.0

    x_train = transform_windows(x_train_raw, feature_scaler)
    x_val = transform_windows(x_val_raw, feature_scaler)
    x_fit = transform_windows(x_fit_raw, feature_scaler)
    x_tune = transform_windows(x_tune_raw, feature_scaler)
    y_train = normalize_target(y_train_real, target_mean, target_std)
    y_val = normalize_target(y_val_real, target_mean, target_std)
    y_fit = normalize_target(y_fit_real, target_mean, target_std)
    y_tune = normalize_target(y_tune_real, target_mean, target_std)

    logger.info(
        "Prepared GEFCom dataset x_train=%s x_val=%s features=%s",
        x_train.shape,
        x_val.shape,
        len(features.columns),
    )
    return RobustPreparedDataset(
        features=features,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_fit=x_fit,
        y_fit=y_fit,
        x_tune=x_tune,
        y_tune=y_tune,
        y_train_real=y_train_real,
        y_fit_real=y_fit_real,
        y_tune_real=y_tune_real,
        y_val_real=y_val_real,
        y_train_sequence=repeat_targets(y_train, window_size),
        y_fit_sequence=repeat_targets(y_fit, window_size),
        y_tune_sequence=repeat_targets(y_tune, window_size),
        y_val_sequence=repeat_targets(y_val, window_size),
        feature_scaler=feature_scaler,
        target_mean=target_mean,
        target_std=target_std,
        feature_names=tuple(features.columns),
        feature_mode=feature_mode,
        metadata=dict(features.attrs.get("metadata", {})),
    )


def create_raw_windows(
    features: pd.DataFrame,
    window_size: int,
    forecast_horizon: int,
    target_feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    if target_feature not in features.columns:
        raise ValueError(f"Target feature not found: {target_feature}")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    values = features.to_numpy(dtype=np.float32)
    target_index = features.columns.get_loc(target_feature)
    sample_count = len(features) - window_size - forecast_horizon + 1
    if sample_count <= 0:
        raise ValueError("Not enough rows for requested window/horizon")

    x = np.empty((sample_count, window_size, values.shape[1]), dtype=np.float32)
    y = np.empty((sample_count,), dtype=np.float32)
    for start in range(sample_count):
        end = start + window_size
        target_row = end + forecast_horizon - 1
        x[start] = values[start:end]
        y[start] = values[target_row, target_index]
    return x, y


def split_windows(
    x: np.ndarray,
    y: np.ndarray,
    validation_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    split_index = int(len(x) * (1 - validation_split))
    if split_index <= 0 or split_index >= len(x):
        raise ValueError("Split leaves an empty partition")
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]


def fit_feature_scaler(x_fit: np.ndarray, columns: pd.Index) -> ZScoreScaler:
    flat = x_fit.reshape(-1, x_fit.shape[-1])
    mean = pd.Series(flat.mean(axis=0), index=columns)
    std = pd.Series(flat.std(axis=0), index=columns).replace(0, 1.0)
    return ZScoreScaler(mean=mean, std=std)


def transform_windows(x: np.ndarray, scaler: ZScoreScaler) -> np.ndarray:
    mean = scaler.mean.to_numpy(dtype=np.float32)
    std = scaler.std.to_numpy(dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def normalize_target(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.asarray(y, dtype=np.float32) - mean) / std).astype(np.float32)


def repeat_targets(y: np.ndarray, sequence_length: int) -> np.ndarray:
    return np.repeat(y.reshape(-1, 1, 1), sequence_length, axis=1).astype(np.float32)
