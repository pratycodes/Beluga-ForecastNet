"""Beluga ForecastNet pipeline using GEFCom features and BWO tuning."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from comparison_models.robust_features import (
    FEATURE_MODE_GEFCOM_LOAD,
    RobustPreparedDataset,
    prepare_robust_dataset,
)
from lfs_hdlbwo.metrics import regression_metrics
from lfs_hdlbwo.model import build_cblstm_ae
from lfs_hdlbwo.optimization import BWO

logger = logging.getLogger(__name__)

CONV_FILTER_OPTIONS = (32, 64, 96, 128)
BILSTM_UNIT_OPTIONS = (64, 96, 128, 160)
DECODER_UNIT_OPTIONS = (32, 64, 96, 128)
DENSE_UNIT_OPTIONS = (16, 32, 64)
LEARNING_RATE_OPTIONS = (0.0001, 0.0003, 0.0005, 0.001, 0.0013, 0.002, 0.003)
BATCH_SIZE_OPTIONS = (32, 64, 128, 160)


@dataclass(frozen=True)
class LfsHdlBwoParams:
    conv_filters: int
    bilstm_units: int
    decoder_units: int
    dense_units: int
    learning_rate: float
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposedRunConfig:
    dataset_path: Path = Path("dataset/GEFCom2014 Data")
    output_dir: Path = Path("artifacts/beluga_forecastnet")
    window_size: int = 72
    forecast_horizon: int = 1
    validation_split: float = 0.2
    tune_split: float = 0.2
    feature_mode: str = FEATURE_MODE_GEFCOM_LOAD
    start_date: str | None = None
    end_date: str | None = None
    recent_years: int | None = 1
    random_seed: int = 42
    bwo_population_size: int = 6
    bwo_max_iter: int = 4
    candidate_epochs: int = 25
    final_epoch_buffer: int = 2
    early_stopping_patience: int = 6
    min_delta: float = 0.0001
    reduce_lr_patience: int = 3
    bwo_patience: int = 0
    epoch_log_interval: int = 1
    save_model: bool = True
    verbose: int = 0
    baseline_metrics_path: Path | None = None


@dataclass(frozen=True)
class ProposedRunResult:
    best_params: LfsHdlBwoParams
    best_tune_rmse: float
    holdout_metrics: dict[str, float]
    output_dir: Path


def run_proposed_lfshdlbwo(config: ProposedRunConfig | None = None) -> ProposedRunResult:
    """Run Beluga ForecastNet with BWO-tuned hyperparameters."""

    config = config or ProposedRunConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", _json_ready(asdict(config)))

    logger.info("Starting Beluga ForecastNet run output_dir=%s", output_dir)
    prepared = prepare_robust_dataset(
        dataset_path=config.dataset_path,
        window_size=config.window_size,
        forecast_horizon=config.forecast_horizon,
        validation_split=config.validation_split,
        tune_split=config.tune_split,
        feature_mode=config.feature_mode,
        start_date=config.start_date,
        end_date=config.end_date,
        recent_years=config.recent_years,
    )
    logger.info(
        "Prepared proposed input feature_mode=%s feature_count=%s",
        prepared.feature_mode,
        len(prepared.feature_names),
    )
    diagnostics = _dataset_diagnostics(prepared, config)
    _write_json(output_dir / "dataset_diagnostics.json", diagnostics)
    logger.info("Dataset signal assessment: %s", diagnostics["viability"])
    logger.info(
        "Prepared tensors train=%s fit=%s tune=%s holdout=%s target_train=%s target_holdout=%s",
        prepared.x_train.shape,
        prepared.x_fit.shape,
        prepared.x_tune.shape,
        prepared.x_val.shape,
        prepared.y_train.shape,
        prepared.y_val.shape,
    )
    logger.info(
        "BWO search budget population=%s iterations=%s max_candidate_evaluations=%s candidate_epochs=%s",
        config.bwo_population_size,
        config.bwo_max_iter,
        config.bwo_population_size * (config.bwo_max_iter + 1),
        config.candidate_epochs,
    )

    trial_rows: list[dict[str, Any]] = []
    fitness = _BwoFitness(prepared=prepared, config=config, output_dir=output_dir, rows=trial_rows)
    optimizer = BWO(
        population_size=config.bwo_population_size,
        max_iter=config.bwo_max_iter,
        dimension=6,
        lower_bound=(0, 0, 0, 0, 0, 0),
        upper_bound=(
            len(CONV_FILTER_OPTIONS) - 1,
            len(BILSTM_UNIT_OPTIONS) - 1,
            len(DECODER_UNIT_OPTIONS) - 1,
            len(DENSE_UNIT_OPTIONS) - 1,
            len(LEARNING_RATE_OPTIONS) - 1,
            len(BATCH_SIZE_OPTIONS) - 1,
        ),
        seed=config.random_seed,
        verbose=True,
        no_improvement_patience=config.bwo_patience or None,
        min_delta=config.min_delta,
    )
    best_position, best_tune_rmse = optimizer.optimize(fitness)
    best_params = decode_proposed_position(best_position)
    best_candidate = fitness.best_candidate_for(best_params)

    _write_json(
        output_dir / "bwo_history.json",
        {
            "best_history": optimizer.best_history,
            "best_position": best_position.tolist(),
            "best_tune_rmse": best_tune_rmse,
        },
    )
    pd.DataFrame(trial_rows).to_csv(output_dir / "bwo_trials.csv", index=False)
    _write_json(output_dir / "best_params.json", best_params.to_dict())
    logger.info("Best BWO params=%s tune_rmse=%.6f", best_params.to_dict(), best_tune_rmse)

    final_epochs = max(1, int(best_candidate.get("best_epoch", 1)) + config.final_epoch_buffer)
    final_result = _fit_final_model(
        prepared=prepared,
        config=config,
        params=best_params,
        final_epochs=final_epochs,
        output_dir=output_dir,
    )
    holdout_metrics = final_result["metrics"]
    _write_json(output_dir / "metrics.json", holdout_metrics)
    _write_json(output_dir / "comparison_to_baselines.json", _compare_to_baselines(config, holdout_metrics))
    _save_plots(output_dir, final_result["y_true"], final_result["y_pred"], final_result["history"])
    logger.info("Beluga ForecastNet holdout metrics=%s", holdout_metrics)

    return ProposedRunResult(
        best_params=best_params,
        best_tune_rmse=float(best_tune_rmse),
        holdout_metrics=holdout_metrics,
        output_dir=output_dir,
    )


class _BwoFitness:
    def __init__(
        self,
        prepared: RobustPreparedDataset,
        config: ProposedRunConfig,
        output_dir: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        self.prepared = prepared
        self.config = config
        self.output_dir = output_dir
        self.rows = rows
        self.cache: dict[str, dict[str, Any]] = {}

    def __call__(self, position: np.ndarray) -> float:
        params = decode_proposed_position(position)
        key = json.dumps(params.to_dict(), sort_keys=True)
        if key in self.cache:
            logger.debug("Using cached BWO candidate params=%s", params.to_dict())
            return float(self.cache[key]["tune_rmse"])

        trial_index = len(self.rows) + 1
        logger.info(
            "Starting BWO candidate %s params=%s fit_shape=%s tune_shape=%s epochs=%s batch_size=%s",
            trial_index,
            params.to_dict(),
            self.prepared.x_fit.shape,
            self.prepared.x_tune.shape,
            self.config.candidate_epochs,
            params.batch_size,
        )
        result = _fit_candidate_model(
            prepared=self.prepared,
            config=self.config,
            params=params,
            trial_index=trial_index,
        )
        row = {
            "trial_index": trial_index,
            "params_json": key,
            **result,
        }
        self.rows.append(row)
        self.cache[key] = row
        pd.DataFrame(self.rows).to_csv(self.output_dir / "bwo_trials_partial.csv", index=False)
        logger.info(
            "BWO candidate %s status=%s tune_rmse=%s epochs=%s seconds=%.2f",
            row["trial_index"],
            row["status"],
            row.get("tune_rmse"),
            row.get("epochs_ran"),
            row.get("total_seconds", np.nan),
        )
        if result["status"] != "ok":
            return float("inf")
        return float(result["tune_rmse"])

    def best_candidate_for(self, params: LfsHdlBwoParams) -> dict[str, Any]:
        key = json.dumps(params.to_dict(), sort_keys=True)
        return self.cache.get(key, {"best_epoch": 1})


def decode_proposed_position(position: np.ndarray) -> LfsHdlBwoParams:
    values = np.asarray(position, dtype=float)
    if values.shape[0] != 6:
        raise ValueError("Proposed BWO position must have six dimensions")
    return LfsHdlBwoParams(
        conv_filters=_select(values[0], CONV_FILTER_OPTIONS),
        bilstm_units=_select(values[1], BILSTM_UNIT_OPTIONS),
        decoder_units=_select(values[2], DECODER_UNIT_OPTIONS),
        dense_units=_select(values[3], DENSE_UNIT_OPTIONS),
        learning_rate=_select(values[4], LEARNING_RATE_OPTIONS),
        batch_size=_select(values[5], BATCH_SIZE_OPTIONS),
    )


def _fit_candidate_model(
    prepared: RobustPreparedDataset,
    config: ProposedRunConfig,
    params: LfsHdlBwoParams,
    trial_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    keras_backend.clear_session()
    tf.keras.utils.set_random_seed(config.random_seed)
    try:
        model = build_cblstm_ae(
            sequence_length=prepared.x_fit.shape[1],
            num_features=prepared.x_fit.shape[2],
            conv_filters=params.conv_filters,
            bilstm_units=params.bilstm_units,
            decoder_units=params.decoder_units,
            dense_units=params.dense_units,
            learning_rate=params.learning_rate,
        )
        history = model.fit(
            prepared.x_fit,
            prepared.y_fit_sequence,
            epochs=config.candidate_epochs,
            batch_size=params.batch_size,
            validation_data=(prepared.x_tune, prepared.y_tune_sequence),
            callbacks=_callbacks(
                config,
                phase=f"BWO candidate {trial_index}",
                total_epochs=config.candidate_epochs,
            ),
            verbose=config.verbose,
        )
        predictions = model.predict(prepared.x_tune, verbose=0)[:, -1, 0]
        y_pred = prepared.inverse_target(predictions)
        metrics = regression_metrics(prepared.y_tune_real, y_pred)
        return {
            "status": "ok",
            "tune_mse": metrics["mse"],
            "tune_rmse": metrics["rmse"],
            "tune_mae": metrics["mae"],
            "tune_mape": metrics["mape"],
            "epochs_ran": len(history.history.get("loss", [])),
            "best_epoch": _best_epoch(history.history),
            "best_val_loss": float(np.nanmin(history.history.get("val_loss", [np.nan]))),
            "total_seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:
        logger.exception("BWO candidate failed params=%s", params.to_dict())
        return {
            "status": "failed",
            "tune_mse": np.nan,
            "tune_rmse": np.nan,
            "tune_mae": np.nan,
            "tune_mape": np.nan,
            "epochs_ran": 0,
            "best_epoch": 0,
            "best_val_loss": np.nan,
            "total_seconds": time.perf_counter() - started,
            "error": str(exc),
        }
    finally:
        keras_backend.clear_session()


def _fit_final_model(
    prepared: RobustPreparedDataset,
    config: ProposedRunConfig,
    params: LfsHdlBwoParams,
    final_epochs: int,
    output_dir: Path,
) -> dict[str, Any]:
    keras_backend.clear_session()
    tf.keras.utils.set_random_seed(config.random_seed)
    logger.info("Final refit on full training split for epochs=%s", final_epochs)
    logger.info(
        "Starting final refit params=%s train_shape=%s holdout_shape=%s batch_size=%s",
        params.to_dict(),
        prepared.x_train.shape,
        prepared.x_val.shape,
        params.batch_size,
    )
    model = build_cblstm_ae(
        sequence_length=prepared.x_train.shape[1],
        num_features=prepared.x_train.shape[2],
        conv_filters=params.conv_filters,
        bilstm_units=params.bilstm_units,
        decoder_units=params.decoder_units,
        dense_units=params.dense_units,
        learning_rate=params.learning_rate,
    )
    history = model.fit(
        prepared.x_train,
        prepared.y_train_sequence,
        epochs=final_epochs,
        batch_size=params.batch_size,
        callbacks=[
            _EpochLogger(
                phase="Final refit",
                total_epochs=final_epochs,
                log_interval=config.epoch_log_interval,
            )
        ],
        verbose=config.verbose,
    )
    predictions = model.predict(prepared.x_val, verbose=0)[:, -1, 0]
    y_pred = prepared.inverse_target(predictions)
    y_true = prepared.y_val_real
    metrics = regression_metrics(y_true, y_pred)
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    if config.save_model:
        model.save(output_dir / "final_model.keras")
    keras_backend.clear_session()
    return {
        "metrics": metrics,
        "history": history.history,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _dataset_diagnostics(
    prepared: RobustPreparedDataset,
    config: ProposedRunConfig,
) -> dict[str, Any]:
    load = prepared.features["load"].to_numpy(dtype=float)
    sample_count = len(load) - config.window_size - config.forecast_horizon + 1
    split_index = int(sample_count * (1 - config.validation_split))
    target_indices = (
        np.arange(sample_count) + config.window_size + config.forecast_horizon - 1
    )
    train_target_indices = target_indices[:split_index]
    holdout_target_indices = target_indices[split_index:]
    y_true = load[holdout_target_indices]

    train_mean = float(np.mean(load[train_target_indices]))
    mean_predictions = np.full_like(y_true, train_mean, dtype=float)
    previous_hour_predictions = load[holdout_target_indices - 1]
    previous_day_indices = np.maximum(holdout_target_indices - 24, 0)
    previous_day_predictions = load[previous_day_indices]

    autocorrelation = {
        str(lag): _safe_correlation(load[lag:], load[:-lag])
        for lag in (1, 2, 3, 6, 12, 24, 48, 168)
        if len(load) > lag
    }
    lead_correlations: list[dict[str, float | str]] = []
    horizon = config.forecast_horizon
    for column in prepared.features.columns:
        feature_values = prepared.features[column].to_numpy(dtype=float)
        correlation = _safe_correlation(feature_values[:-horizon], load[horizon:])
        lead_correlations.append({"feature": column, "correlation": correlation})
    lead_correlations = sorted(
        lead_correlations,
        key=lambda item: abs(float(item["correlation"])),
        reverse=True,
    )

    target_derived_inputs = [
        feature
        for feature in prepared.feature_names
        if feature in {"normalized_consumption", "energy_efficiency_score"}
    ]
    max_autocorrelation = max((abs(value) for value in autocorrelation.values()), default=0.0)
    max_lead_correlation = max(
        (
            abs(float(item["correlation"]))
            for item in lead_correlations
            if item["feature"] != "load"
        ),
        default=0.0,
    )
    mean_metrics = regression_metrics(y_true, mean_predictions)
    previous_hour_metrics = regression_metrics(y_true, previous_hour_predictions)
    previous_day_metrics = regression_metrics(y_true, previous_day_predictions)

    issues = []
    if max_autocorrelation < 0.1:
        issues.append("target autocorrelation is very weak")
    if max_lead_correlation < 0.1:
        issues.append("non-target input lead correlations are very weak")
    if previous_hour_metrics["rmse"] > mean_metrics["rmse"]:
        issues.append("previous-hour persistence is worse than the train mean")
    if previous_day_metrics["rmse"] > mean_metrics["rmse"]:
        issues.append("previous-day persistence is worse than the train mean")
    if target_derived_inputs:
        issues.append("target-derived input columns are present")

    viability = "viable"
    if len(issues) >= 3:
        viability = "weak_signal_not_recommended"
    elif issues:
        viability = "limited_signal"

    return {
        "feature_mode": prepared.feature_mode,
        "dataset_metadata": prepared.metadata,
        "feature_count": len(prepared.feature_names),
        "row_count": int(len(prepared.features)),
        "sample_count": int(sample_count),
        "train_sample_count": int(len(train_target_indices)),
        "holdout_sample_count": int(len(holdout_target_indices)),
        "window_size": config.window_size,
        "forecast_horizon": config.forecast_horizon,
        "train_mean_baseline": mean_metrics,
        "previous_hour_baseline": previous_hour_metrics,
        "previous_day_baseline": previous_day_metrics,
        "autocorrelation": autocorrelation,
        "top_lead_feature_correlations": lead_correlations[:12],
        "target_derived_inputs": target_derived_inputs,
        "max_abs_autocorrelation": max_autocorrelation,
        "max_abs_non_target_lead_correlation": max_lead_correlation,
        "issues": issues,
        "viability": viability,
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _callbacks(config: ProposedRunConfig, phase: str, total_epochs: int):
    return [
        _EpochLogger(
            phase=phase,
            total_epochs=total_epochs,
            log_interval=config.epoch_log_interval,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_delta=config.min_delta,
            min_lr=1e-6,
        ),
        TerminateOnNaN(),
    ]


class _EpochLogger(Callback):
    """Log Keras epoch progress through the project logger."""

    def __init__(self, phase: str, total_epochs: int, log_interval: int) -> None:
        super().__init__()
        self.phase = phase
        self.total_epochs = total_epochs
        self.log_interval = max(1, int(log_interval))
        self.started_at = 0.0
        self.epoch_started_at = 0.0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        _ = logs
        self.started_at = time.perf_counter()
        logger.info("%s training started total_epochs=%s", self.phase, self.total_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        _ = logs
        self.epoch_started_at = time.perf_counter()
        if self._should_log(epoch):
            logger.info("%s epoch %s/%s started", self.phase, epoch + 1, self.total_epochs)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        if not self._should_log(epoch):
            return
        logger.info(
            "%s epoch %s/%s finished loss=%s mae=%s rmse=%s val_loss=%s val_mae=%s val_rmse=%s lr=%s epoch_seconds=%.2f total_seconds=%.2f",
            self.phase,
            epoch + 1,
            self.total_epochs,
            _format_metric(logs.get("loss")),
            _format_metric(logs.get("mae")),
            _format_metric(logs.get("rmse")),
            _format_metric(logs.get("val_loss")),
            _format_metric(logs.get("val_mae")),
            _format_metric(logs.get("val_rmse")),
            _format_metric(_current_learning_rate(self.model)),
            time.perf_counter() - self.epoch_started_at,
            time.perf_counter() - self.started_at,
        )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        _ = logs
        logger.info("%s training finished seconds=%.2f", self.phase, time.perf_counter() - self.started_at)

    def _should_log(self, epoch: int) -> bool:
        return epoch == 0 or (epoch + 1) % self.log_interval == 0 or epoch + 1 == self.total_epochs


def _current_learning_rate(model: Any) -> float | None:
    try:
        value = model.optimizer.learning_rate
        if callable(value):
            value = value(model.optimizer.iterations)
        return float(keras_backend.get_value(value))
    except Exception:
        return None


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.6f}"


def _compare_to_baselines(
    config: ProposedRunConfig,
    holdout_metrics: dict[str, float],
) -> dict[str, Any]:
    path = config.baseline_metrics_path
    if path is None or not Path(path).exists():
        return {"baseline_metrics_path": str(path) if path else None, "available": False}

    baselines = pd.read_csv(path)
    rmse_column = _first_existing_column(
        baselines,
        ("holdout_rmse", "holdout_rmse_mean", "rmse"),
    )
    if rmse_column is None:
        return {
            "baseline_metrics_path": str(path),
            "available": False,
            "error": "No supported RMSE column found in baseline metrics file",
        }
    best_idx = baselines[rmse_column].idxmin()
    best_baseline = baselines.loc[best_idx].to_dict()
    proposed_rmse = holdout_metrics["rmse"]
    best_baseline_rmse = float(best_baseline[rmse_column])
    return {
        "baseline_metrics_path": str(path),
        "available": True,
        "best_baseline": best_baseline,
        "proposed_rmse": proposed_rmse,
        "best_baseline_rmse": best_baseline_rmse,
        "rmse_delta_vs_best_baseline": proposed_rmse - best_baseline_rmse,
        "beats_best_baseline": proposed_rmse < best_baseline_rmse,
    }


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _save_plots(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: dict[str, list[float]],
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Validation sample")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(y_pred - y_true, bins=30)
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=150)
    plt.close()


def _best_epoch(history: dict[str, list[float]]) -> int:
    val_loss = history.get("val_loss", [])
    if not val_loss:
        return 1
    return int(np.nanargmin(val_loss) + 1)


def _select(value: float, options: tuple[Any, ...]) -> Any:
    index = int(np.rint(value))
    index = int(np.clip(index, 0, len(options) - 1))
    return options[index]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
