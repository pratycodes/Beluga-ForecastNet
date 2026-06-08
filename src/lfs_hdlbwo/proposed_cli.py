"""CLI for the Beluga ForecastNet forecasting pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from comparison_models.robust_features import FEATURE_MODES
from lfs_hdlbwo.proposed import ProposedRunConfig, run_proposed_lfshdlbwo


def build_parser() -> argparse.ArgumentParser:
    defaults = ProposedRunConfig()
    parser = argparse.ArgumentParser(
        description="Run Beluga ForecastNet with BWO hyperparameter tuning."
    )
    parser.add_argument("--dataset-path", type=Path, default=defaults.dataset_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--forecast-horizon", type=int, default=defaults.forecast_horizon)
    parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
    parser.add_argument("--tune-split", type=float, default=defaults.tune_split)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=defaults.feature_mode)
    parser.add_argument("--start-date", type=str, default=defaults.start_date)
    parser.add_argument("--end-date", type=str, default=defaults.end_date)
    parser.add_argument(
        "--recent-years",
        type=int,
        default=defaults.recent_years,
        help="For GEFCom mode, use the most recent N complete years; use 0 for all known rows.",
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--bwo-population-size", type=int, default=defaults.bwo_population_size)
    parser.add_argument("--bwo-max-iter", type=int, default=defaults.bwo_max_iter)
    parser.add_argument("--candidate-epochs", type=int, default=defaults.candidate_epochs)
    parser.add_argument(
        "--bwo-patience",
        type=int,
        default=defaults.bwo_patience,
        help="BWO no-improvement patience; use 0 to disable BWO early stopping.",
    )
    parser.add_argument(
        "--epoch-log-interval",
        type=int,
        default=defaults.epoch_log_interval,
        help="Log Keras training metrics every N epochs; default logs every epoch.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--final-epoch-buffer", type=int, default=defaults.final_epoch_buffer)
    parser.add_argument("--baseline-metrics-path", type=Path, default=defaults.baseline_metrics_path)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--verbose", type=int, default=defaults.verbose)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick BWO wiring check with tiny search settings.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = ProposedRunConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        validation_split=args.validation_split,
        tune_split=args.tune_split,
        feature_mode=args.feature_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        recent_years=None if args.recent_years == 0 else args.recent_years,
        random_seed=args.seed,
        bwo_population_size=args.bwo_population_size,
        bwo_max_iter=args.bwo_max_iter,
        candidate_epochs=args.candidate_epochs,
        bwo_patience=args.bwo_patience,
        epoch_log_interval=args.epoch_log_interval,
        early_stopping_patience=args.early_stopping_patience,
        final_epoch_buffer=args.final_epoch_buffer,
        baseline_metrics_path=args.baseline_metrics_path,
        save_model=not args.no_save_model,
        verbose=args.verbose,
    )
    if args.smoke:
        config = replace(
            config,
            window_size=24,
            start_date="2014-01-01",
            end_date="2014-01-31",
            bwo_population_size=2,
            bwo_max_iter=1,
            candidate_epochs=1,
            early_stopping_patience=1,
            bwo_patience=1,
            save_model=False,
        )

    result = run_proposed_lfshdlbwo(config)
    print("Best params:", result.best_params.to_dict())
    print("Best tune RMSE:", result.best_tune_rmse)
    print("Holdout metrics:", result.holdout_metrics)
    print("Artifacts:", result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
