"""Contrastive losses on L2-normalised embeddings.

InfoNCE (user -> shop): for each user anchor in the batch, the shop rows with the same match_key are
positives and every other shop row is a negative. Multi-positive form:
    L_i = -log( sum_{p in P(i)} exp(s_ip / T) / sum_{j in shop} exp(s_ij / T) )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_user_to_shop(emb: torch.Tensor, keys: torch.Tensor, is_user: torch.Tensor, temperature: float) -> torch.Tensor:
    """emb (B,D) normalised; keys (B,) int labels; is_user (B,) bool."""
    a, s = emb[is_user], emb[~is_user]
    logits = a @ s.T / temperature                                   # (Na, Ns)
    pos = keys[is_user][:, None] == keys[~is_user][None, :]          # (Na, Ns) bool
    log_all = torch.logsumexp(logits, dim=1)
    log_pos = torch.logsumexp(logits.masked_fill(~pos, float("-inf")), dim=1)
    return (log_all - log_pos).mean()


def arcface_logits(emb: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, scale: float, margin: float) -> torch.Tensor:
    """ArcFace: cosine logits with an additive angular margin on the true class (weight rows L2-normalised)."""
    cos = emb @ F.normalize(weight, dim=1).T
    theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    target = torch.cos(theta + margin)
    onehot = F.one_hot(labels, cos.shape[1]).bool()
    return scale * torch.where(onehot, target, cos)
