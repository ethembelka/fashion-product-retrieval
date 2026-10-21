"""Thin CLI: python scripts/finetune.py --processed <dir with crops/ splits/> --out <checkpoint dir> --features <features root>

Colab: --processed /content/processed --out <Drive>/outputs/checkpoints/ft --features <Drive>/features/siglip_vitb16+ft
Smoke test: --smoke  (64 products, 2 epochs, tiny val)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fashret  # noqa: E402,F401  (must precede torch: see fashret/__init__.py)
import torch  # noqa: E402

from fashret.config import PathsCfg, load_yaml  # noqa: E402
from fashret.engine.finetune import finetune  # noqa: E402
from fashret.utils.seed import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train/finetune.yaml"))
    ap.add_argument("--backbones", default=str(ROOT / "configs/models/backbones.yaml"))
    ap.add_argument("--processed", default=None)
    ap.add_argument("--out", default=None, help="checkpoint dir (Drive on Colab)")
    ap.add_argument("--features", default=None, help="features root for the fine-tuned model")
    ap.add_argument("--splits", nargs="+", default=["val_internal", "test"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    spec = load_yaml(args.backbones)["backbones"][cfg["backbone"]]
    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    processed = Path(args.processed) if args.processed else paths.processed_dir
    out = Path(args.out) if args.out else paths.outputs_dir / "checkpoints" / "ft"
    features = Path(args.features) if args.features else paths.features_dir / f"{cfg['backbone']}+ft"
    init = cfg["head"].get("init_from")
    if init and not Path(init).is_absolute():
        cfg["head"]["init_from"] = str(ROOT / init)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lim, vlim = (64, 400) if args.smoke else (None, None)
    if args.smoke:
        cfg["optim"]["epochs"], cfg["optim"]["warmup_steps"], cfg["num_workers"] = 2, 2, 2
    print(f"device={device} processed={processed} out={out} features={features}")
    s = finetune(cfg, spec, processed, out, features, args.splits, device, limit_products=lim, val_limit=vlim)
    print(f"best val R@1 = {s['best_val_recall@1']:.4f} after {s['epochs_run']} epochs")


if __name__ == "__main__":
    main()
