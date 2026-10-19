"""Pack data/processed/crops + splits into data/crops.zip for Google Drive / Colab.

JPEGs are already compressed, so entries are stored (no deflate) — much faster to write and read.
Usage: python scripts/pack_crops.py [--out data/crops.zip]
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashret.config import DataCfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/crops.zip"))
    args = ap.parse_args()
    cfg = DataCfg.from_yaml(ROOT / "configs/data.yaml", root=ROOT)
    processed = cfg.paths.processed_dir
    files = sorted((processed / "crops").rglob("*.jpg")) + sorted((processed / "splits").rglob("*.csv"))
    files += [processed / "crop_report.json"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in tqdm(files, desc="zip", mininterval=5):
            zf.write(p, p.relative_to(processed))
            total += p.stat().st_size
    print(f"{len(files)} files, {total/1e9:.2f} GB raw, zip {out.stat().st_size/1e9:.2f} GB, {time.time()-t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
