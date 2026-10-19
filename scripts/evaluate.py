"""Thin CLI: python scripts/evaluate.py --features data/features/siglip_vitb16/test [--config configs/eval/default.yaml]

For the test split, τ is read from the newest val_internal run of the same model/variant (protocol:
τ is chosen on val_internal only). Writes outputs/results/<run_id>/{config.yaml, git_hash.txt,
metrics.json, per_query.parquet, per_openset.parquet, coverage_curve_{abs,margin}.csv, breakdown.csv}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashret.config import PathsCfg, load_yaml  # noqa: E402
from fashret.evaluation.evaluator import evaluate_split  # noqa: E402
from fashret.utils.io import write_json, write_table  # noqa: E402
from fashret.utils.run import init_run_dir, make_run_id  # noqa: E402


def find_tau(results_dir: Path, model: str, variant: str, split: str) -> dict[str, float] | None:
    runs = sorted(results_dir.glob(f"*_{model}_{variant}_{split}_s*/metrics.json"))
    if not runs:
        return None
    m = json.loads(runs[-1].read_text(encoding="utf-8"))
    return m.get("openset", {}).get("tau")


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
    oc = cfg["openset"]

    tau = None
    if split != oc["tau_select_split"]:
        tau = find_tau(paths.outputs_dir / "results", model, args.variant, oc["tau_select_split"])
        if tau is None:
            sys.exit(f"No {oc['tau_select_split']} run for {model}/{args.variant}; evaluate that split first (τ is chosen there).")

    res = evaluate_split(feat_dir, cfg["recall_k"], cfg["search_k"], tau=tau,
                         tau_target_coverage=oc["tau_target_coverage"], report_coverages=tuple(oc["report_coverages"]))
    metrics = res["metrics"]
    metrics.update(model=model, split=split, variant=args.variant)

    run_id = make_run_id(model, f"{args.variant}_{split}", args.seed)
    run_dir = init_run_dir(paths.outputs_dir, run_id, Path(args.config))
    write_json(metrics, run_dir / "metrics.json")
    write_table(res["per_query"], run_dir / "per_query.parquet")
    write_table(res["per_openset"], run_dir / "per_openset.parquet")
    for name, curve in res["curves"].items():
        curve.to_csv(run_dir / f"coverage_curve_{name}.csv", index=False)
    res["breakdown"].to_csv(run_dir / "breakdown.csv", index=False)

    o = metrics["openset"]
    print(f"run_id: {run_id}")
    print("  " + "  ".join(f"R@{k}={metrics[f'recall@{k}']:.3f}" for k in cfg["recall_k"]) + f"  mAP={metrics['map']:.3f}")
    print(f"  AUROC abs={o['auroc_abs']:.3f}  margin={o['auroc_margin']:.3f}   (τ from {o['tau_source']})")
    for n in ("abs", "margin"):
        t = o[f"at_tau_{n}"]
        print(f"  τ_{n}={t['tau']:.3f}: coverage={t['coverage']:.3f} precision={t['precision']:.3f} false_match={t['false_match_rate']:.3f}")


if __name__ == "__main__":
    main()
