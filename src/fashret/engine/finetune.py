"""Partial fine-tuning of the SigLIP image tower (last N blocks) + projection head, from crop images.

Per step: P products × (1 user + 1 shop) crops -> backbone -> head -> InfoNCE user->shop.
Per epoch: val_internal Recall@1 with the current model (features recomputed), best/last checkpoints
(trainable params only) written to `out_dir` — on Colab that is a Drive folder, so a lost session resumes.
At the end: features for the requested splits are extracted with the best model (same layout as extractor.py).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from ..data.samplers import PKSampler
from ..evaluation.closed_set import hits_matrix, recall_at_k
from ..index.faiss_index import FlatIPIndex
from ..models.base import l2_normalize
from ..models.projection_head import ProjectionHead
from ..utils.io import ensure_dir, write_json, write_table
from .losses import infonce_user_to_shop


class FineTuneModel(nn.Module):
    """open_clip SigLIP image tower + MLP head; only the last `unfreeze_blocks` blocks, attn_pool, norm and head train."""

    def __init__(self, arch: str, pretrained: str, unfreeze_blocks: int, head_cfg: dict) -> None:
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = model.visual
        self.eval_tf = preprocess
        trunk = self.visual.trunk
        for p in self.visual.parameters():
            p.requires_grad_(False)
        train_mods = list(trunk.blocks[len(trunk.blocks) - unfreeze_blocks:]) + [trunk.norm, trunk.attn_pool]
        for m in train_mods:
            for p in m.parameters():
                p.requires_grad_(True)
        with torch.no_grad():
            d = self.visual(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.head = ProjectionHead(d, head_cfg["hidden_dim"], head_cfg["out_dim"], head_cfg["dropout"])
        self.backbone_dim = d

    def backbone_params(self):
        return [p for p in self.visual.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(l2_normalize(self.visual(x)))

    def trainable_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if k.startswith("head.") or _is_trainable_key(self, k)}


def _is_trainable_key(model: FineTuneModel, key: str) -> bool:
    for name, p in model.named_parameters():
        if name == key:
            return p.requires_grad
    return False


class CropImages(Dataset):
    def __init__(self, items: pd.DataFrame, processed: Path, tf) -> None:
        self.paths = [processed / p for p in items["crop_path"]]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), i


def train_transform(eval_tf, scale: tuple[float, float], hflip: bool):
    norm = eval_tf.transforms[-1]
    ops = [T.RandomResizedCrop(224, scale=tuple(scale), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    return T.Compose(ops + [T.ToTensor(), T.Normalize(norm.mean, norm.std)])


@torch.no_grad()
def encode_items(model: FineTuneModel, items: pd.DataFrame, processed: Path, device, bs: int, workers: int, amp: bool, desc: str) -> np.ndarray:
    model.eval()
    dl = DataLoader(CropImages(items, processed, model.eval_tf), batch_size=bs, num_workers=workers, pin_memory=True)
    out = np.zeros((len(items), model.head.out_dim), dtype=np.float32)
    for x, idx in tqdm(dl, desc=desc, mininterval=5, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z = model(x.to(device, non_blocking=True))
        out[idx.numpy()] = z.float().cpu().numpy()
    return out


def val_recall(model, processed, val_lists: dict[str, pd.DataFrame], device, bs, workers, amp) -> dict[str, float]:
    g = encode_items(model, val_lists["gallery"], processed, device, bs, workers, amp, "val gallery")
    q = encode_items(model, val_lists["queries"], processed, device, bs, workers, amp, "val queries")
    idx = FlatIPIndex(g.shape[1]); idx.add(g)
    _, I = idx.search(q, 10)
    return recall_at_k(hits_matrix(val_lists["gallery"]["match_key"].to_numpy()[I], val_lists["queries"]["match_key"].to_numpy()), (1, 5, 10))


def finetune(cfg: dict, backbone_spec: dict, processed: Path, out_dir: Path, features_out: Path | None, extract_splits: list[str],
             device: torch.device, limit_products: int | None = None, val_limit: int | None = None) -> dict:
    out_dir = ensure_dir(out_dir)
    splits = processed / "splits"
    tr_g, tr_q = pd.read_csv(splits / "train_fit/gallery.csv"), pd.read_csv(splits / "train_fit/queries.csv")
    items = pd.concat([tr_q, tr_g], ignore_index=True)
    is_user = np.r_[np.ones(len(tr_q), bool), np.zeros(len(tr_g), bool)]
    keys = items["match_key"].to_numpy()
    if limit_products:  # smoke test
        keep = set(pd.unique(tr_q["match_key"])[:limit_products])
        mask = items["match_key"].isin(keep).to_numpy()
        items, is_user, keys = items[mask].reset_index(drop=True), is_user[mask], keys[mask]
    val_lists = {n: pd.read_csv(splits / cfg["val_split"] / f"{n}.csv") for n in ("gallery", "queries")}
    if val_limit:
        val_lists = {n: d.head(val_limit) for n, d in val_lists.items()}
        val_lists["queries"] = val_lists["queries"][val_lists["queries"]["match_key"].isin(set(val_lists["gallery"]["match_key"]))]

    model = FineTuneModel(backbone_spec["name"], backbone_spec["pretrained"], cfg["unfreeze_blocks"], cfg["head"]).to(device)
    init = Path(cfg["head"].get("init_from", "")) if cfg["head"].get("init_from") else None
    if init and init.exists():
        ck = torch.load(init, map_location=device)
        model.head.load_state_dict(ck["head"]); print(f"head initialised from {init}")
    n_bb = sum(p.numel() for p in model.backbone_params()); n_head = sum(p.numel() for p in model.head.parameters())
    print(f"trainable: backbone {n_bb/1e6:.1f}M (last {cfg['unfreeze_blocks']} blocks + pool), head {n_head/1e6:.1f}M")

    oc, sc, lc = cfg["optim"], cfg["sampler"], cfg["loss"]
    opt = torch.optim.AdamW([{"params": model.backbone_params(), "lr": oc["lr_backbone"]},
                             {"params": model.head.parameters(), "lr": oc["lr_head"]}], weight_decay=oc["weight_decay"])
    sampler = PKSampler(keys, is_user, sc["P"], sc["K"], seed=cfg["seed"])
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * oc["epochs"]
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int) -> None:  # linear warmup then cosine
        if step < oc["warmup_steps"]:
            f = (step + 1) / oc["warmup_steps"]
        else:
            f = 0.5 * (1 + math.cos(math.pi * (step - oc["warmup_steps"]) / max(1, total_steps - oc["warmup_steps"])))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    tf = train_transform(model.eval_tf, cfg["augment"]["random_resized_crop_scale"], cfg["augment"]["hflip"])
    ds = CropImages(items, processed, tf)
    amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    Y = torch.from_numpy(np.unique(keys, return_inverse=True)[1]).to(device)
    U = torch.from_numpy(is_user).to(device)
    eval_bs, workers = 256, cfg["num_workers"]

    history, best, bad, start_epoch, step = [], -1.0, 0, 0, 0
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        history, best, bad, start_epoch, step = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1, ck["step"]
        print(f"resumed from epoch {ck['epoch']} (best val R@1 {best:.4f})")

    for epoch in range(start_epoch, oc["epochs"]):
        model.train(); t0 = time.time(); losses = []
        dl = DataLoader(ds, batch_sampler=list(sampler), num_workers=workers, pin_memory=True, persistent_workers=False)
        for x, idx in tqdm(dl, desc=f"epoch {epoch}", mininterval=10, leave=False):
            set_lr(step)
            b = idx.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                z = model(x.to(device, non_blocking=True))
                loss = infonce_user_to_shop(z.float(), Y[b], U[b], lc["temperature"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            losses.append(loss.item()); step += 1
        r = val_recall(model, processed, val_lists, device, eval_bs, workers, amp)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in r.items()}, "seconds": round(time.time() - t0, 1), "step": step}
        history.append(rec); print(json.dumps(rec), flush=True)
        if r["recall@1"] > best:
            best, bad = r["recall@1"], 0
            torch.save({"model": model.trainable_state(), "cfg": cfg, "backbone": backbone_spec, "epoch": epoch, "val": r}, out_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "history": history,
                    "best": best, "bad": bad, "epoch": epoch, "step": step}, last)
        write_json(history, out_dir / "history.json")
        if bad >= oc["patience"]:
            print(f"early stop at epoch {epoch}"); break

    summary = {"best_val_recall@1": best, "epochs_run": len(history), "n_train_rows": int(len(items)), "n_products": int(len(np.unique(keys))),
               "trainable_backbone_params": n_bb, "trainable_head_params": n_head, "history": history}
    write_json(summary, out_dir / "train_summary.json")

    if features_out and extract_splits:
        ck = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        for split in extract_splits:
            dst = ensure_dir(features_out / split)
            for lst in ("gallery", "queries", "openset"):
                li = pd.read_csv(splits / split / f"{lst}.csv")
                emb = encode_items(model, li, processed, device, eval_bs, workers, amp, f"{split}/{lst}")
                np.save(dst / f"{lst}_emb.npy", emb); write_table(li, dst / f"{lst}_meta.parquet")
            write_json({"model": f"siglip_vitb16+ft", "split": split, "embed_dim": int(model.head.out_dim), "lists": {}}, dst / "extract_report.json")
            print(f"features -> {dst}", flush=True)
    return summary
