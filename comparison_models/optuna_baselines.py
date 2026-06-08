"""Optuna tuning for baseline comparison models only."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from comparison_models.deep_learning import (
    build_bilstm_baseline,
    build_cblstm_ae_baseline,
    build_cnn_bilstm_baseline,
    build_cnn_lstm_baseline,
    build_gru_baseline,
    build_lstm_baseline,
)
from comparison_models.robust_features import (
    FEATURE_MODE_GEFCOM_LOAD,
    RobustPreparedDataset,
    prepare_robust_dataset,
)
from comparison_models.statistical import fit_arima_forecast, fit_sarima_forecast
from lfs_hdlbwo.metrics import regression_metrics

logger = logging.getLogger(__name__)

OPTUNA_BASELINE_MODELS = (
    "arima",
    "sarima",
    "lstm",
    "gru",
    "bilstm",
    "cnn_lstm",
    "cnn_bilstm",
    "cblstm_ae",
)
OPTUNA_DEEP_MODELS = (
    "lstm",
    "gru",
    "bilstm",
    "cnn_lstm",
    "cnn_bilstm",
    "cblstm_ae",
)


@dataclass(frozen=True)
class OptunaBaselineConfig:
    dataset_path: Path = Path("dataset/GEFCom2014 Data")
    output_dir: Path = Path("artifacts/optuna_baseline_comparisons")
    seed: int = 42
    n_trials: int = 8
    statistical_trials: int = 6
    max_epochs: int = 30
    patience: int = 6
    min_delta: float = 0.0001
    reduce_lr_patience: int = 3
    validation_split: float = 0.2
    tune_split: float = 0.2
    window_size: int = 72
    forecast_horizon: int = 1
    feature_mode: str = FEATURE_MODE_GEFCOM_LOAD
    start_date: str | None = None
    end_date: str | None = None
    recent_years: int | None = 1
    statistical_train_limit: int = 2000
    save_models: bool = False
    verbose: int = 0


def run_optuna_baseline_comparison(
    config: OptunaBaselineConfig | None = None,
    model_names: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Tune baseline models with Optuna and evaluate on the holdout split."""

    config = config or OptunaBaselineConfig()
    requested = _normalize_model_names(model_names or OPTUNA_BASELINE_MODELS)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "optuna_config.json", _config_payload(config, requested))

    logger.info("Running Optuna baseline comparison models=%s", requested)
    logger.info("Seed=%s output_dir=%s", config.seed, output_dir)
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

    trial_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for model_name in requested:
        started = time.perf_counter()
        logger.info("Starting Optuna tuning for %s", model_name)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.seed),
            study_name=f"{model_name}_seed_{config.seed}",
        )
        objective = _objective_for_model(model_name, config, prepared, trial_rows)
        trials = config.n_trials if model_name in OPTUNA_DEEP_MODELS else config.statistical_trials
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
        model_trials = [row for row in trial_rows if row["model"] == model_name]
        best_row = _final_evaluate_best(
            model_name=model_name,
            config=config,
            prepared=prepared,
            study=study,
            output_dir=output_dir,
            tuning_seconds=time.perf_counter() - started,
        )
        best_rows.append(best_row)
        _save_progress(output_dir, trial_rows, best_rows)
        _write_json(
            output_dir / f"{model_name}_study_trials.json",
            model_trials,
        )

    trials_df = pd.DataFrame(trial_rows)
    best_df = pd.DataFrame(best_rows).sort_values("holdout_rmse").reset_index(drop=True)
    _save_results(output_dir, trials_df, best_df)
    logger.info("Finished Optuna baseline comparison. Results saved to %s", output_dir)
    return {
        "optuna_trials": trials_df,
        "comparison_metrics": best_df,
    }


def _objective_for_model(
    model_name: str,
    config: OptunaBaselineConfig,
    prepared: RobustPreparedDataset,
    trial_rows: list[dict[str, Any]],
):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(model_name, trial)
        row_started = time.perf_counter()
        if model_name in OPTUNA_DEEP_MODELS:
            result = _fit_deep_once(
                model_name=model_name,
                config=config,
                prepared=prepared,
                params=params,
                fit_partition="fit",
                predict_partition="tune",
                output_dir=None,
            )
        else:
            result = _fit_statistical_once(
                model_name=model_name,
                config=config,
                train_series=prepared.y_fit_real,
                target_series=prepared.y_tune_real,
                params=params,
            )
        row = {
            "model": model_name,
            "seed": config.seed,
            "trial_number": trial.number,
            "params_json": json.dumps(params, sort_keys=True),
            "status": result["status"],
            "tune_mse": result["mse"],
            "tune_rmse": result["rmse"],
            "tune_mae": result["mae"],
            "tune_mape": result["mape"],
            "fit_seconds": result["fit_seconds"],
            "predict_seconds": result["predict_seconds"],
            "total_seconds": time.perf_counter() - row_started,
            "epochs_ran": result.get("epochs_ran", 0),
            "best_epoch": result.get("best_epoch", 0),
            "error": result.get("error", ""),
        }
        trial_rows.append(row)
        if result["status"] != "ok":
            return float("inf")
        trial.set_user_attr("metrics", row)
        return float(result["rmse"])

    return objective


def _final_evaluate_best(
    model_name: str,
    config: OptunaBaselineConfig,
    prepared: RobustPreparedDataset,
    study: optuna.Study,
    output_dir: Path,
    tuning_seconds: float,
) -> dict[str, Any]:
    params = dict(study.best_trial.params)
    logger.info("Best %s params=%s tune_rmse=%.6f", model_name, params, study.best_value)
    if model_name in OPTUNA_DEEP_MODELS:
        best_epoch = int(
            study.best_trial.user_attrs.get("metrics", {}).get("best_epoch")
            or config.max_epochs
        )
        result = _fit_deep_once(
            model_name=model_name,
            config=config,
            prepared=prepared,
            params=params,
            fit_partition="train",
            predict_partition="holdout",
            output_dir=output_dir,
            final_epochs=best_epoch,
        )
    else:
        result = _fit_statistical_once(
            model_name=model_name,
            config=config,
            train_series=prepared.y_train_real,
            target_series=prepared.y_val_real,
            params=params,
        )
    return {
        "model": model_name,
        "seed": config.seed,
        "best_trial_number": study.best_trial.number,
        "best_params_json": json.dumps(params, sort_keys=True),
        "best_tune_rmse": float(study.best_value),
        "holdout_mse": result["mse"],
        "holdout_rmse": result["rmse"],
        "holdout_mae": result["mae"],
        "holdout_mape": result["mape"],
        "tuning_seconds": tuning_seconds,
        "final_fit_seconds": result["fit_seconds"],
        "final_predict_seconds": result["predict_seconds"],
        "final_total_seconds": result["total_seconds"],
        "epochs_ran": result.get("epochs_ran", 0),
        "best_epoch": result.get("best_epoch", 0),
        "status": result["status"],
        "error": result.get("error", ""),
    }


def _fit_deep_once(
    model_name: str,
    config: OptunaBaselineConfig,
    prepared: RobustPreparedDataset,
    params: dict[str, Any],
    fit_partition: str,
    predict_partition: str,
    output_dir: Path | None,
    final_epochs: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    keras_backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    try:
        x_fit, y_fit, x_val, y_val = _deep_fit_data(model_name, prepared, fit_partition)
        model = _build_deep_model(
            model_name,
            sequence_length=x_fit.shape[1],
            num_features=x_fit.shape[2],
            params=params,
        )
        callbacks = _callbacks(config)
        epochs = int(final_epochs or config.max_epochs)
        validation_data = None if fit_partition == "train" else (x_val, y_val)
        fit_callbacks = [] if validation_data is None else callbacks
        fit_started = time.perf_counter()
        history = model.fit(
            x_fit,
            y_fit,
            epochs=epochs,
            batch_size=int(params["batch_size"]),
            validation_data=validation_data,
            callbacks=fit_callbacks,
            verbose=config.verbose,
        )
        fit_seconds = time.perf_counter() - fit_started

        x_predict, y_true_real = _predict_data(prepared, predict_partition)
        predict_started = time.perf_counter()
        predictions = model.predict(x_predict, verbose=0)
        predict_seconds = time.perf_counter() - predict_started
        y_pred_scaled = _prediction_to_target(model_name, predictions)
        y_pred_real = prepared.inverse_target(y_pred_scaled)
        metrics = regression_metrics(y_true_real, y_pred_real)

        if output_dir is not None:
            pd.DataFrame(history.history).to_csv(
                output_dir / f"{model_name}_optuna_best_history.csv",
                index=False,
            )
            if config.save_models:
                model.save(output_dir / f"{model_name}_optuna_best.keras")

        return {
            "status": "ok",
            **metrics,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": time.perf_counter() - started,
            "epochs_ran": len(history.history.get("loss", [])),
            "best_epoch": int(final_epochs or _best_epoch(history.history)),
            "error": "",
        }
    except Exception as exc:
        logger.exception("Deep Optuna trial failed model=%s params=%s", model_name, params)
        return _failed_result(started, str(exc))
    finally:
        keras_backend.clear_session()


def _fit_statistical_once(
    model_name: str,
    config: OptunaBaselineConfig,
    train_series: np.ndarray,
    target_series: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    train_series = _last_n(train_series, config.statistical_train_limit)
    try:
        fit_started = time.perf_counter()
        if model_name == "arima":
            forecast = fit_arima_forecast(
                train_series,
                forecast_steps=len(target_series),
                order=(int(params["p"]), int(params["d"]), int(params["q"])),
            )
        elif model_name == "sarima":
            forecast = fit_sarima_forecast(
                train_series,
                forecast_steps=len(target_series),
                order=(int(params["p"]), int(params["d"]), int(params["q"])),
                seasonal_order=(
                    int(params["sp"]),
                    int(params["sd"]),
                    int(params["sq"]),
                    int(params["seasonal_period"]),
                ),
            )
        else:
            raise ValueError(f"Unsupported statistical model: {model_name}")
        fit_seconds = time.perf_counter() - fit_started
        metrics = regression_metrics(target_series, forecast.predictions)
        return {
            "status": "ok",
            **metrics,
            "fit_seconds": fit_seconds,
            "predict_seconds": 0.0,
            "total_seconds": time.perf_counter() - started,
            "epochs_ran": 0,
            "best_epoch": 0,
            "error": "",
        }
    except Exception as exc:
        logger.exception("Statistical Optuna trial failed model=%s params=%s", model_name, params)
        return _failed_result(started, str(exc))


def _suggest_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    if model_name == "arima":
        return {
            "p": trial.suggest_int("p", 0, 3),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 3),
        }
    if model_name == "sarima":
        return {
            "p": trial.suggest_int("p", 0, 2),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 2),
            "sp": trial.suggest_int("sp", 0, 1),
            "sd": trial.suggest_int("sd", 0, 1),
            "sq": trial.suggest_int("sq", 0, 1),
            "seasonal_period": trial.suggest_categorical("seasonal_period", [24]),
        }
    if model_name in ("lstm", "gru", "bilstm"):
        return {
            "units": trial.suggest_categorical("units", [32, 64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35, step=0.05),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    if model_name == "cnn_lstm":
        return {
            "conv_filters": trial.suggest_categorical("conv_filters", [32, 64, 96]),
            "lstm_units": trial.suggest_categorical("lstm_units", [32, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35, step=0.05),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    if model_name == "cnn_bilstm":
        return {
            "conv_filters": trial.suggest_categorical("conv_filters", [32, 64, 96]),
            "bilstm_units": trial.suggest_categorical("bilstm_units", [32, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35, step=0.05),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    if model_name == "cblstm_ae":
        return {
            "conv_filters": trial.suggest_categorical("conv_filters", [32, 64, 96]),
            "bilstm_units": trial.suggest_categorical("bilstm_units", [64, 96, 128]),
            "decoder_units": trial.suggest_categorical("decoder_units", [32, 64, 96]),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _build_deep_model(
    model_name: str,
    sequence_length: int,
    num_features: int,
    params: dict[str, Any],
):
    if model_name == "lstm":
        return build_lstm_baseline(
            sequence_length,
            num_features,
            units=int(params["units"]),
            dropout=float(params["dropout"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    if model_name == "gru":
        return build_gru_baseline(
            sequence_length,
            num_features,
            units=int(params["units"]),
            dropout=float(params["dropout"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    if model_name == "bilstm":
        return build_bilstm_baseline(
            sequence_length,
            num_features,
            units=int(params["units"]),
            dropout=float(params["dropout"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    if model_name == "cnn_lstm":
        return build_cnn_lstm_baseline(
            sequence_length,
            num_features,
            conv_filters=int(params["conv_filters"]),
            lstm_units=int(params["lstm_units"]),
            dropout=float(params["dropout"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    if model_name == "cnn_bilstm":
        return build_cnn_bilstm_baseline(
            sequence_length,
            num_features,
            conv_filters=int(params["conv_filters"]),
            bilstm_units=int(params["bilstm_units"]),
            dropout=float(params["dropout"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    if model_name == "cblstm_ae":
        return build_cblstm_ae_baseline(
            sequence_length,
            num_features,
            conv_filters=int(params["conv_filters"]),
            bilstm_units=int(params["bilstm_units"]),
            decoder_units=int(params["decoder_units"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
        )
    raise ValueError(f"Unsupported deep model: {model_name}")


def _deep_fit_data(
    model_name: str,
    prepared: RobustPreparedDataset,
    fit_partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fit_partition == "fit":
        x_fit = prepared.x_fit
        y_fit = prepared.y_fit_sequence if model_name == "cblstm_ae" else prepared.y_fit
        x_val = prepared.x_tune
        y_val = prepared.y_tune_sequence if model_name == "cblstm_ae" else prepared.y_tune
        return x_fit, y_fit, x_val, y_val
    if fit_partition == "train":
        x_fit = prepared.x_train
        y_fit = prepared.y_train_sequence if model_name == "cblstm_ae" else prepared.y_train
        x_val = prepared.x_tune
        y_val = prepared.y_tune_sequence if model_name == "cblstm_ae" else prepared.y_tune
        return x_fit, y_fit, x_val, y_val
    raise ValueError(f"Unsupported fit partition: {fit_partition}")


def _predict_data(
    prepared: RobustPreparedDataset,
    predict_partition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if predict_partition == "tune":
        return prepared.x_tune, prepared.y_tune_real
    if predict_partition == "holdout":
        return prepared.x_val, prepared.y_val_real
    raise ValueError(f"Unsupported predict partition: {predict_partition}")


def _prediction_to_target(model_name: str, predictions: np.ndarray) -> np.ndarray:
    if model_name == "cblstm_ae":
        return predictions[:, -1, 0]
    return predictions.reshape(-1)


def _callbacks(config: OptunaBaselineConfig):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
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


def _best_epoch(history: dict[str, list[float]]) -> int:
    val_loss = history.get("val_loss", [])
    if not val_loss:
        return 0
    return int(np.nanargmin(val_loss) + 1)


def _last_n(values: np.ndarray, limit: int) -> np.ndarray:
    if limit > 0 and len(values) > limit:
        return values[-limit:]
    return values


def _failed_result(started: float, error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "mse": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "fit_seconds": np.nan,
        "predict_seconds": np.nan,
        "total_seconds": time.perf_counter() - started,
        "epochs_ran": 0,
        "best_epoch": 0,
        "error": error,
    }


def _save_progress(
    output_dir: Path,
    trial_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(trial_rows).to_csv(output_dir / "optuna_trials_partial.csv", index=False)
    pd.DataFrame(best_rows).to_csv(output_dir / "comparison_metrics_partial.csv", index=False)


def _save_results(
    output_dir: Path,
    trials_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    best_df.to_csv(output_dir / "comparison_metrics.csv", index=False)
    _write_json(output_dir / "optuna_trials.json", trials_df.to_dict(orient="records"))
    _write_json(output_dir / "comparison_metrics.json", best_df.to_dict(orient="records"))


def _config_payload(
    config: OptunaBaselineConfig,
    requested: tuple[str, ...],
) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_path"] = str(config.dataset_path)
    payload["output_dir"] = str(config.output_dir)
    payload["model_names"] = list(requested)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_model_names(model_names: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(name.strip().lower().replace("-", "_") for name in model_names)
    unsupported = sorted(set(normalized) - set(OPTUNA_BASELINE_MODELS))
    if unsupported:
        raise ValueError(
            f"Unsupported model names: {unsupported}. "
            f"Supported models: {', '.join(OPTUNA_BASELINE_MODELS)}"
        )
    return normalized
