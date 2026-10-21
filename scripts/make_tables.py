"""Thin CLI: python scripts/make_tables.py  -> reports/tables/{all_runs.csv, table1_backbones.csv, table1_backbones.tex}"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashret.config import PathsCfg  # noqa: E402
from fashret.reporting.tables import backbone_table, latest_per_key, load_runs, to_latex  # noqa: E402


def main() -> None:
    paths = PathsCfg.from_yaml(ROOT / "configs/paths.yaml", ROOT)
    df = latest_per_key(load_runs(paths.outputs_dir / "results"))
    out = ROOT / "reports/tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    for split in ("test", "val_internal"):
        t = backbone_table(df, split=split)
        t.to_csv(out / f"table1_backbones_{split}.csv", index=False)
        if split == "test":
            (out / "table1_backbones.tex").write_text(
                to_latex(t, "Training-free backbone comparison on the DeepFashion2 test split (official val lists).", "tab:backbones"),
                encoding="utf-8")
        print(f"\n[{split}]\n", t.to_string(index=False))
    # Table 2: best backbone, zeroshot vs learned heads (test)
    t2 = df[(df.split == "test") & (df.model.str.startswith("siglip_vitb16"))].sort_values("variant")
    cols = ["model", "variant", "embed_dim"] + [c for c in t2.columns if c.startswith("recall@") or c in ("map", "category_consistency@5", "auroc_abs", "auroc_margin")]
    t2[cols].to_csv(out / "table2_head_test.csv", index=False)
    print("\n[table2 test]\n", t2[cols].to_string(index=False))


if __name__ == "__main__":
    main()
