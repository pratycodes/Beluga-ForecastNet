# Documentation Assets

This directory contains curated outputs for the repository README.

- `figures/`: architecture diagrams, optimization flow, metric plots, and model diagnostic plots
- `results/`: compact result tables and experiment summaries generated from checked experiment artifacts

Regenerate the report assets after new experiments:

```bash
python3 scripts/generate_results_report.py
```

Runtime experiment outputs belong in `artifacts/`, which is ignored by Git. Commit only stable figures and summary tables that are meant to appear in the project documentation.
