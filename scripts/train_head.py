"""Thin CLI: python scripts/train_head.py [--config configs/train/head.yaml] [--variant head] [--resume]

1. trains the head on data/features/<backbone>/train_fit, selecting on val_internal
2. applies best.pt to val_internal and test features -> data/features/<backbone>+<variant>/<split>/
3. writes outputs/checkpoints/<run_id>/{best.pt,last.pt,history.json} and outputs/results/<run_id>/train_summary.json
Then run scripts/evaluate.py --features data/features/<backbone>+<variant>/<split> --variant <variant>.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fashret  # noqa: E402,F401  (must precede torch: see fashret/__init__.py)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fashret.config import PathsCfg, load_yaml  # noqa: E402
from fashret.engine.trainer import apply_head, load_head, train_head  # noqa: E402
from fashret.utils.io import write_json  # noqa: E402
from fashret.utils.run import init_run_dir, make_run_id  # noqa: E402
from fashret.utils.seed import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train/head.yaml"))
    ap.add_argument("--variant", default="head")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["val_internal", "test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = cfg["backbone"]
    feat = paths.features_dir / bb
    run_id = make_run_id(bb, f"{args.variant}_train", cfg["seed"])
    ck_dir = paths.outputs_dir / "checkpoints" / run_id
    run_dir = init_run_dir(paths.outputs_dir, run_id, Path(args.config))

    print(f"device={device} backbone={bb} loss={cfg['loss']['name']} run={run_id}")
    summary = train_head(feat / "train_fit", feat / cfg["val_split"], cfg, ck_dir, device, resume=args.resume)
    write_json(summary, run_dir / "train_summary.json")

    head = load_head(ck_dir / "best.pt", device)
    out_root = paths.features_dir / f"{bb}+{args.variant}"
    for split in args.splits:
        src, dst = feat / split, out_root / split
        dst.mkdir(parents=True, exist_ok=True)
        for lst in ("gallery", "queries", "openset"):
            np.save(dst / f"{lst}_emb.npy", apply_head(head, np.load(src / f"{lst}_emb.npy"), device))
            shutil.copy(src / f"{lst}_meta.parquet", dst / f"{lst}_meta.parquet")
        print(f"  wrote {dst}")
    print(f"best val R@1 = {summary['best_val_recall@1']:.4f} after {summary['epochs_run']} epochs")


if __name__ == "__main__":
    main()
