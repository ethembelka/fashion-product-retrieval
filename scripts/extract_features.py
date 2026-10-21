"""Thin CLI: python scripts/extract_features.py --model dinov2_vitb14 --split test [--lists gallery queries openset]

Paths come from configs/paths.yaml; --processed / --features override them (Colab: /content/crops, Drive).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fashret  # noqa: E402,F401  (must precede torch: see fashret/__init__.py)
import torch  # noqa: E402

from fashret.config import PathsCfg  # noqa: E402
from fashret.engine.extractor import extract_split  # noqa: E402
from fashret.models.registry import build_extractor, load_backbone_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, choices=["train_fit", "val_internal", "test"])
    ap.add_argument("--lists", nargs="+", default=["gallery", "queries", "openset"])
    ap.add_argument("--config", default=str(ROOT / "configs/models/backbones.yaml"))
    ap.add_argument("--processed", default=None, help="override processed_dir (contains crops/ and splits/)")
    ap.add_argument("--features", default=None, help="override features_dir")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only first N items per list")
    args = ap.parse_args()

    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    processed = Path(args.processed) if args.processed else paths.processed_dir
    features = Path(args.features) if args.features else paths.features_dir
    cfg = load_backbone_cfg(args.config)
    spec = cfg["backbones"][args.model]
    bs = args.batch_size or spec.get("batch_size", 64)
    nw = args.workers if args.workers is not None else cfg.get("num_workers", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} split={args.split} batch={bs} workers={nw}")
    model = build_extractor(args.model, cfg, device)
    report = extract_split(model, args.split, args.lists, processed, features, bs, nw, args.overwrite, args.limit)
    for name, st in report["lists"].items():
        print(f"  {name}: {st}")


if __name__ == "__main__":
    main()
