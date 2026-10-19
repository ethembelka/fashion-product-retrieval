"""Matplotlib figures for reports. One palette, thin marks, direct labels, no rainbow.

Palette: fixed categorical order (blue, orange, aqua), one sequential hue (blue) for
single-series magnitude. Ink/grid colours are muted so the data dominates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def hbar(labels: Sequence[str], values: Sequence[float], path: Path, title: str, xlabel: str = "",
         color: str = SERIES[0], fmt: str = "{:,.0f}") -> Path:
    """Single-series horizontal bars, sorted as given, value labelled at the bar end."""
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(labels) + 1.2))
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=color, height=0.6)
    ax.set_yticks(list(y), labels)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v).replace(",", "."), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, vmax * 1.12)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    _style(ax)
    return _save(fig, path)


def grouped_hbar(groups: Sequence[str], series: dict[str, Sequence[float]], path: Path, title: str,
                 fmt: str = "{:,.0f}") -> Path:
    """Few series x few groups; fixed hue order; legend always present."""
    n = len(series)
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(groups) + 1.4))
    h = 0.8 / n
    vmax = max(max(s) for s in series.values())
    for i, (name, vals) in enumerate(series.items()):
        ys = [g * 1.0 + (n - 1 - i) * h for g in range(len(groups))][::-1]
        ax.barh(ys, vals, height=h * 0.9, color=SERIES[i], label=name)
        for yy, v in zip(ys, vals):
            ax.text(v + vmax * 0.01, yy, fmt.format(v).replace(",", "."), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, vmax * 1.15)
    centers = [g * 1.0 + (n - 1) * h / 2 for g in range(len(groups))][::-1]
    ax.set_yticks(centers, groups)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    _style(ax)
    return _save(fig, path)


def box_and_crops(image_path: str, boxes: Sequence[tuple[int, int, int, int, str]], crop_paths: Sequence[str],
                  path: Path, title: str) -> Path:
    """Left: original photo with labelled boxes. Right: the crops produced from them."""
    k = len(crop_paths)
    fig = plt.figure(figsize=(3.2 + 2.0 * k, 3.6))
    gs = fig.add_gridspec(1, 1 + k, width_ratios=[1.6] + [1] * k)
    ax = fig.add_subplot(gs[0, 0])
    with Image.open(image_path) as im:
        ax.imshow(im)
    for i, (x1, y1, x2, y2, lab) in enumerate(boxes):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, edgecolor=SERIES[i % len(SERIES)]))
        ax.text(x1 + 4, y1 + 4, lab, fontsize=7.5, color="white", weight="bold", va="top",
                bbox=dict(facecolor=SERIES[i % len(SERIES)], edgecolor="none", pad=1.5))
    ax.set_title("Orijinal fotoğraf + kutular", fontsize=9, color=INK2)
    ax.axis("off")
    for i, cp in enumerate(crop_paths):
        axc = fig.add_subplot(gs[0, 1 + i])
        with Image.open(cp) as c:
            axc.imshow(c)
        axc.set_title(f"Kırpıntı {i + 1}", fontsize=9, color=INK2)
        axc.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    return _save(fig, path)


def retrieval_grid(rows: Sequence[tuple[Path, Sequence[tuple[Path, bool, float]]]], path: Path, title: str) -> Path:
    """Each row: query crop, then its top-k gallery crops; correct ones get a green frame, score under each."""
    k = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), k + 1, figsize=(1.55 * (k + 1), 1.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for ri, (qp, results) in enumerate(rows):
        ax = axes[ri, 0]
        with Image.open(qp) as im:
            ax.imshow(im)
        ax.set_title("sorgu" if ri == 0 else "", fontsize=8, color=INK2)
        for s in ax.spines.values():
            s.set_edgecolor(SERIES[0]); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        for ci in range(k):
            ax = axes[ri, ci + 1]
            if ci < len(results):
                gp, ok, score = results[ci]
                with Image.open(gp) as im:
                    ax.imshow(im)
                col = "#0ca30c" if ok else GRID
                for s in ax.spines.values():
                    s.set_edgecolor(col); s.set_linewidth(3 if ok else 1)
                ax.set_xlabel(f"{score:.2f}", fontsize=7, color=INK2)
                if ri == 0:
                    ax.set_title(f"top-{ci + 1}", fontsize=8, color=INK2)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def flow_diagram(steps: Sequence[str], path: Path, title: str) -> Path:
    """Left-to-right boxes with arrows; for 'how the data moves' figures."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(1.9 * n + 0.5, 1.6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(steps):
        ax.add_patch(Rectangle((i + 0.08, 0.25), 0.84, 0.5, facecolor="#e8f0fb", edgecolor=SERIES[0], lw=1.2))
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=8.5, color=INK, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(i + 1.08, 0.5), xytext=(i + 0.92, 0.5),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return _save(fig, path)
