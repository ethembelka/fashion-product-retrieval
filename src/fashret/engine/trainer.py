"""Projection-head training on cached backbone embeddings (backbone frozen).

Data: train_fit gallery (shop) + queries (user) embeddings and their match_keys. Each epoch: P×K batches,
InfoNCE user->shop (or ArcFace over product classes). After each epoch: val_internal Recall@1 with the
head applied to val_internal gallery/queries. Keeps best.pt (by val R@1) and last.pt (resume).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.samplers import PKSampler
from ..evaluation.closed_set import hits_matrix, recall_at_k
from ..index.faiss_index import FlatIPIndex
from ..models.projection_head import ProjectionHead
from ..utils.io import ensure_dir, read_table
from .losses import arcface_logits, infonce_user_to_shop


def load_split(feat_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Returns (emb, keys, is_user, gallery_meta, queries_meta) for one split's gallery + queries."""
    g = np.load(feat_dir / "gallery_emb.npy"); gm = read_table(feat_dir / "gallery_meta.parquet")
    q = np.load(feat_dir / "queries_emb.npy"); qm = read_table(feat_dir / "queries_meta.parquet")
    emb = np.concatenate([q, g]).astype(np.float32)
    keys = np.concatenate([qm["match_key"].to_numpy(), gm["match_key"].to_numpy()])
    is_user = np.r_[np.ones(len(q), bool), np.zeros(len(g), bool)]
    return emb, keys, is_user, gm, qm


@torch.no_grad()
def apply_head(head: nn.Module, emb: np.ndarray, device: torch.device, bs: int = 8192) -> np.ndarray:
    head.eval()
    out = [head(torch.from_numpy(emb[i:i + bs]).to(device)).cpu().numpy() for i in range(0, len(emb), bs)]
    return np.concatenate(out).astype(np.float32)


def val_recall(head: nn.Module, val_dir: Path, device: torch.device, ks=(1, 5, 10)) -> dict[str, float]:
    g = apply_head(head, np.load(val_dir / "gallery_emb.npy"), device)
    q = apply_head(head, np.load(val_dir / "queries_emb.npy"), device)
    gk = read_table(val_dir / "gallery_meta.parquet")["match_key"].to_numpy()
    qk = read_table(val_dir / "queries_meta.parquet")["match_key"].to_numpy()
    idx = FlatIPIndex(g.shape[1]); idx.add(g)
    _, I = idx.search(q, max(ks))
    return recall_at_k(hits_matrix(gk[I], qk), ks)


def train_head(train_dir: Path, val_dir: Path, cfg: dict, out_dir: Path, device: torch.device, resume: bool = False) -> dict:
    ensure_dir(out_dir)
    emb, keys, is_user, _, _ = load_split(train_dir)
    key_ids, key_int = np.unique(keys, return_inverse=True)
    X = torch.from_numpy(emb).to(device)
    Y = torch.from_numpy(key_int).to(device)
    U = torch.from_numpy(is_user).to(device)

    hc, lc, oc, sc = cfg["head"], cfg["loss"], cfg["optim"], cfg["sampler"]
    head = ProjectionHead(emb.shape[1], hc["hidden_dim"], hc["out_dim"], hc["dropout"]).to(device)
    groups = [{"params": list(head.parameters()), "lr": oc["lr"]}]
    arc_w = None
    if lc["name"] == "arcface":
        # one class-centre row per product; each row sees ~2 samples per epoch, so it needs its own (higher) lr
        arc_w = nn.Parameter(torch.randn(len(key_ids), hc["out_dim"], device=device) * 0.01)
        groups.append({"params": [arc_w], "lr": lc.get("class_lr", oc["lr"]), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=oc["lr"], weight_decay=oc["weight_decay"])
    sampler = PKSampler(keys, is_user, sc["P"], sc["K"], seed=cfg["seed"])

    history, best, bad, start_epoch = [], -1.0, 0, 0
    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        history, best, bad, start_epoch = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        if arc_w is not None: arc_w.data.copy_(ck["arc_w"])

    for epoch in range(start_epoch, oc["epochs"]):
        head.train(); t0 = time.time(); losses = []
        for batch in sampler:
            b = torch.from_numpy(batch).to(device)
            z = head(X[b])
            if arc_w is None:
                loss = infonce_user_to_shop(z, Y[b], U[b], lc["temperature"])
            else:
                loss = nn.functional.cross_entropy(arcface_logits(z, arc_w, Y[b], lc["scale"], lc["margin"]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        r = val_recall(head, val_dir, device)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in r.items()}, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        if r["recall@1"] > best:
            best, bad = r["recall@1"], 0
            torch.save({"head": head.state_dict(), "cfg": cfg, "in_dim": emb.shape[1], "epoch": epoch, "val": r}, out_dir / "best.pt")
        else:
            bad += 1
        ck = {"head": head.state_dict(), "opt": opt.state_dict(), "history": history, "best": best, "bad": bad, "epoch": epoch}
        if arc_w is not None: ck["arc_w"] = arc_w.data
        torch.save(ck, last)
        if bad >= oc["patience"]:
            print(f"early stop at epoch {epoch} (best val R@1 {best:.4f})"); break

    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return {"best_val_recall@1": best, "epochs_run": len(history), "n_train_rows": int(len(emb)), "n_products": int(len(key_ids)), "history": history}


def load_head(path: Path, device: torch.device) -> ProjectionHead:
    ck = torch.load(path, map_location=device)
    hc = ck["cfg"]["head"]
    head = ProjectionHead(ck["in_dim"], hc["hidden_dim"], hc["out_dim"], hc["dropout"]).to(device)
    head.load_state_dict(ck["head"]); head.eval()
    return head
