"""Thin CLI: python scripts/evaluate.py --features data/features/dinov2_vitb14/test [--config configs/eval/default.yaml]

Writes outputs/results/<run_id>/{config.yaml, git_hash.txt, metrics.json, per_query.parquet}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashret.config import PathsCfg, load_yaml  # noqa: E402
from fashret.evaluation.evaluator import evaluate_closed_set  # noqa: E402
from fashret.utils.io import write_json, write_table  # noqa: E402
from fashret.utils.run import init_run_dir, make_run_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir with gallery_/queries_/openset_ emb+meta")
    ap.add_argument("--config", default=str(ROOT / "configs/eval/default.yaml"))
    ap.add_argument("--variant", default="zeroshot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    feat_dir = Path(args.features)
    model, split = feat_dir.parent.name, feat_dir.name

    metrics, per_query, eff = evaluate_closed_set(feat_dir, cfg["recall_k"], cfg["search_k"])
    metrics["efficiency"] = eff
    metrics["model"], metrics["split"], metrics["variant"] = model, split, args.variant

    run_id = make_run_id(model, f"{args.variant}_{split}", args.seed)
    run_dir = init_run_dir(paths.outputs_dir, run_id, Path(args.config))
    write_json(metrics, run_dir / "metrics.json")
    write_table(per_query, run_dir / "per_query.parquet")

    print(f"run_id: {run_id}")
    for k in [f"recall@{k}" for k in cfg["recall_k"]] + ["map"]:
        print(f"  {k:10s} {metrics[k]:.4f}")
    print(f"  cat_cons@5 {metrics['category_consistency@5']:.4f}  dim={eff['embed_dim']}  {eff['search_ms_per_query']:.3f} ms/q")


if __name__ == "__main__":
    main()
