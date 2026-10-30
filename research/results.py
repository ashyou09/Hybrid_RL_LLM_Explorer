"""Run folders: every run writes its config, metadata and metrics to disk.

Layout::

    results/<level>/runs/<timestamp>_<name>/
        config.json   – the exact knobs
        meta.json     – command, git commit, Python version, time
        metrics.json  – numbers produced by the run
        ...           – plots, replays, tables
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from research.config import Config

RESULTS_ROOT = Path("results")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def new_run(cfg: Config, level: str = "level_1", root: Path = RESULTS_ROOT) -> Path:
    """Create a run folder and write ``config.json`` and ``meta.json`` into it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = root / level / "runs" / f"{stamp}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    meta = {
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "started_utc": stamp,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def save_metrics(run_dir: Path, metrics: dict) -> Path:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path
