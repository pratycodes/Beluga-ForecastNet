"""Baseline comparison models for the LFS-HDLBWO workflow."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_bilstm_baseline",
    "build_cblstm_ae_baseline",
    "build_cnn_bilstm_baseline",
    "build_cnn_lstm_baseline",
    "build_gru_baseline",
    "build_lstm_baseline",
    "run_optuna_baseline_comparison",
]


def __getattr__(name: str) -> Any:
    if name in {
        "build_bilstm_baseline",
        "build_cblstm_ae_baseline",
        "build_cnn_bilstm_baseline",
        "build_cnn_lstm_baseline",
        "build_gru_baseline",
        "build_lstm_baseline",
    }:
        deep_learning = importlib.import_module("comparison_models.deep_learning")

        return getattr(deep_learning, name)
    if name == "run_optuna_baseline_comparison":
        from comparison_models.optuna_baselines import run_optuna_baseline_comparison

        return run_optuna_baseline_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
