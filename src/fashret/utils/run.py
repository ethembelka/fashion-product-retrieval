"""Run bookkeeping: run_id, git hash, config snapshot under outputs/results/<run_id>/."""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path

from .io import ensure_dir


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # not a git checkout (e.g. Colab copy)
        return "unknown"


def make_run_id(model: str, variant: str, seed: int = 0, day: date | None = None) -> str:
    day = day or date.today()
    return f"{day:%Y%m%d}_{model}_{variant}_s{seed}"


def init_run_dir(outputs_dir: Path, run_id: str, config_path: Path | None = None) -> Path:
    run_dir = ensure_dir(outputs_dir / "results" / run_id)
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
    if config_path is not None:
        shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir
