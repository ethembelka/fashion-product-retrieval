"""Thin CLI: python scripts/prepare_data.py [--config configs/data.yaml] [--workers 8]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashret.config import DataCfg  # noqa: E402
from fashret.data.prepare import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = DataCfg.from_yaml(args.config, root=ROOT)
    out = prepare(cfg, workers=args.workers)
    print("\nCrop report:", out["crop_report"])
    print("\nSummary:\n", out["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
