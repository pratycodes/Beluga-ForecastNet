"""Generate compact result tables and figures for the README."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import pandas as pd


PROPOSED_DIR = PROJECT_ROOT / "artifacts" / "beluga_forecastnet"
LEGACY_PROPOSED_DIR = PROJECT_ROOT / "artifacts" / "lfshdlbwo_proposed"
BASELINE_DIR = PROJECT_ROOT / "artifacts" / "optuna_baseline_comparisons"
ASSET_PROPOSED_DIR = PROJECT_ROOT / "assets" / "lfshdlbwo_proposed"
ASSET_BASELINE_DIR = PROJECT_ROOT / "assets" / "optuna_baseline_comparisons"
DOCS_RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
PROPOSED_MODEL_NAME = "Beluga ForecastNet"


def main() -> int:
    DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    proposed_dir = _existing_dir(
        PROPOSED_DIR,
        LEGACY_PROPOSED_DIR,
        ASSET_PROPOSED_DIR,
    )
    baseline_dir = _existing_dir(BASELINE_DIR, ASSET_BASELINE_DIR)
    proposed_metrics = _read_json(proposed_dir / "metrics.json")
    proposed_config = _read_json(proposed_dir / "run_config.json")
    proposed_diagnostics = _read_json(proposed_dir / "dataset_diagnostics.json")
    baseline_metrics = pd.read_csv(baseline_dir / "comparison_metrics.csv")

    comparison = _build_comparison_table(proposed_metrics, baseline_metrics)
    comparison.to_csv(DOCS_RESULTS_DIR / "model_comparison.csv", index=False)
    _write_json(
        DOCS_RESULTS_DIR / "experiment_summary.json",
        {
            "proposed_metrics": proposed_metrics,
            "proposed_config": proposed_config,
            "dataset": proposed_diagnostics.get("dataset_metadata", {}),
            "proposed_rank_by_rmse": int(
                comparison.loc[
                    comparison["model"].eq(PROPOSED_MODEL_NAME),
                    "rank_by_rmse",
                ].iloc[0]
            ),
            "models_compared": comparison["model"].tolist(),
            "seed_count": 1,
            "robustness_status": "single_seed_current_artifacts",
        },
    )

    _plot_metric_bar(
        comparison,
        metric="rmse",
        ylabel="RMSE",
        path=DOCS_FIGURES_DIR / "rmse_comparison.png",
    )
    _plot_metric_bar(
        comparison,
        metric="mape",
        ylabel="MAPE (%)",
        path=DOCS_FIGURES_DIR / "mape_comparison.png",
    )
    _plot_bwo_trials(proposed_dir / "bwo_trials.csv", DOCS_FIGURES_DIR / "bwo_candidate_rmse.png")
    _plot_training_loss(proposed_dir / "training_history.csv", DOCS_FIGURES_DIR / "proposed_training_loss.png")
    _copy_if_exists(proposed_dir / "actual_vs_predicted.png", DOCS_FIGURES_DIR / "proposed_actual_vs_predicted.png")
    _copy_if_exists(proposed_dir / "error_distribution.png", DOCS_FIGURES_DIR / "proposed_error_distribution.png")

    print(f"Wrote {DOCS_RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Wrote figures to {DOCS_FIGURES_DIR}")
    return 0


def _build_comparison_table(
    proposed_metrics: dict[str, float],
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "model": PROPOSED_MODEL_NAME,
            "rmse": float(proposed_metrics["rmse"]),
            "mae": float(proposed_metrics["mae"]),
            "mape": float(proposed_metrics["mape"]),
        }
    ]
    for row in baseline_metrics.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "rmse": float(row.holdout_rmse),
                "mae": float(row.holdout_mae),
                "mape": float(row.holdout_mape),
            }
        )
    frame = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    frame.insert(0, "rank_by_rmse", frame.index + 1)
    return frame


def _plot_metric_bar(frame: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    ordered = frame.sort_values(metric, ascending=True)
    colors = ["#2457a6" if model == PROPOSED_MODEL_NAME else "#8d99ae" for model in ordered["model"]]
    plt.figure(figsize=(10, 5.5))
    plt.bar(ordered["model"], ordered[metric], color=colors)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.title(f"Model comparison by {ylabel}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bwo_trials(path: Path, output_path: Path) -> None:
    trials = pd.read_csv(path).sort_values("trial_index")
    plt.figure(figsize=(8, 4.5))
    plt.plot(trials["trial_index"], trials["tune_rmse"], marker="o", color="#2457a6")
    plt.xlabel("BWO candidate")
    plt.ylabel("Tune RMSE")
    plt.title("BWO candidate search")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_training_loss(path: Path, output_path: Path) -> None:
    history = pd.read_csv(path)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(history) + 1), history["loss"], marker="o", color="#2457a6")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Final Beluga ForecastNet training loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copyfile(source, destination)


def _existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"None of these directories exists: {joined}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
