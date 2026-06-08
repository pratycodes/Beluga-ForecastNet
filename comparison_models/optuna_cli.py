"""CLI for one-seed Optuna baseline comparisons."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from comparison_models.robust_features import FEATURE_MODES
from comparison_models.optuna_baselines import (
    OPTUNA_BASELINE_MODELS,
    OptunaBaselineConfig,
    run_optuna_baseline_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = OptunaBaselineConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna tuning for baseline comparison models only. "
            "This does not run LFS-HDLBWO."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=defaults.dataset_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--models", nargs="+", default=list(OPTUNA_BASELINE_MODELS))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-trials", type=int, default=defaults.n_trials)
    parser.add_argument("--statistical-trials", type=int, default=defaults.statistical_trials)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=defaults.reduce_lr_patience,
    )
    parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
    parser.add_argument("--tune-split", type=float, default=defaults.tune_split)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--forecast-horizon", type=int, default=defaults.forecast_horizon)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=defaults.feature_mode)
    parser.add_argument("--start-date", type=str, default=defaults.start_date)
    parser.add_argument("--end-date", type=str, default=defaults.end_date)
    parser.add_argument(
        "--recent-years",
        type=int,
        default=defaults.recent_years,
        help="For GEFCom mode, use the most recent N complete years; use 0 for all known rows.",
    )
    parser.add_argument(
        "--statistical-train-limit",
        type=int,
        default=defaults.statistical_train_limit,
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--verbose", type=int, default=defaults.verbose)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one quick GRU trial for wiring validation.",
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
    config = OptunaBaselineConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        statistical_trials=args.statistical_trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        reduce_lr_patience=args.reduce_lr_patience,
        validation_split=args.validation_split,
        tune_split=args.tune_split,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        feature_mode=args.feature_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        recent_years=None if args.recent_years == 0 else args.recent_years,
        statistical_train_limit=args.statistical_train_limit,
        save_models=args.save_models,
        verbose=args.verbose,
    )
    models = tuple(args.models)
    if args.smoke:
        config = replace(
            config,
            window_size=24,
            start_date="2014-01-01",
            end_date="2014-01-31",
            n_trials=1,
            statistical_trials=1,
            max_epochs=1,
            patience=1,
        )
        if args.models == list(OPTUNA_BASELINE_MODELS):
            models = ("gru",)

    frames = run_optuna_baseline_comparison(config=config, model_names=models)
    print(frames["comparison_metrics"].to_string(index=False))
    print(f"Artifacts: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
