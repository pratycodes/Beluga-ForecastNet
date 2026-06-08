"""Compatibility wrapper for running Beluga ForecastNet."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfs_hdlbwo.proposed_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
