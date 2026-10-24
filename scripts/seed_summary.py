"""Aggregate seed repeats: mean ± std per method on the test split.

Groups runs by method family: 'zeroshot' (1 run), 'head' (head, head_s1, head_s2, ...), 'arcface', 'ft' (ft, ft_s1, ...).
Writes reports/tables/seed_summary.csv (+ .tex) and reports/figures/seed_summary_r1.png.
Usage: python scripts/seed_summary.py [--split test]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fashret.config import PathsCfg  # noqa: E402
from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.tables import load_runs  # noqa: E402

METRICS = ["recall@1", "recall@5", "recall@10", "recall@20", "map", "category_consistency@5", "auroc_abs", "auroc_margin"]
LABEL = {"zeroshot": "Eğitimsiz SigLIP", "head": "+ başlık (InfoNCE)", "arcface": "+ başlık (ArcFace)", "ft": "+ kısmi fine-tune"}


def family(variant: str) -> str:
    return re.sub(r"_s\d+$", "", variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    df = load_runs(paths.outputs_dir / "results")
    df = df[(df.split == args.split) & df.model.str.startswith("siglip_vitb16")].copy()
    df["family"] = df.variant.map(family)
    df["seed"] = df.run_id.str.extract(r"_s(\d+)$").astype(int)
    df = df.sort_values("run_id").groupby(["family", "seed"], as_index=False).last()   # newest run per (family, seed)

    rows = []
    for fam in ("zeroshot", "head", "arcface", "ft"):
        g = df[df.family == fam]
        if g.empty:
            continue
        row = {"method": LABEL[fam], "family": fam, "n_seeds": len(g), "seeds": ",".join(map(str, sorted(g.seed)))}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    tdir = ROOT / "reports/tables"; tdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tdir / f"seed_summary_{args.split}.csv", index=False)

    disp = out[["method", "n_seeds"]].copy()
    for m in METRICS[:5]:
        disp[m] = [f"{a:.3f} ± {s:.3f}" if n > 1 else f"{a:.3f}" for a, s, n in zip(out[f"{m}_mean"], out[f"{m}_std"], out.n_seeds)]
    print(disp.to_string(index=False))
    (tdir / f"seed_summary_{args.split}.tex").write_text(disp.to_latex(index=False, escape=True), encoding="utf-8")
    fig.hbar_err(list(out.method), list(out["recall@1_mean"]), list(out["recall@1_std"]), ROOT / "reports/figures" / f"seed_summary_r1_{args.split}.png",
                 f"Recall@1 — tohum ortalaması ± standart sapma ({args.split})", "Recall@1")
    # per-run detail for the report
    detail = df[["family", "seed", "run_id"] + METRICS].sort_values(["family", "seed"])
    detail.to_csv(tdir / f"seed_runs_{args.split}.csv", index=False)
    print("\n", detail[["family", "seed", "recall@1", "map", "auroc_abs"]].to_string(index=False))


if __name__ == "__main__":
    main()
