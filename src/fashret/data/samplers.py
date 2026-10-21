"""P×K batch sampler over cached embeddings: P products per batch, K images each (>=1 user + >=1 shop)."""

from __future__ import annotations

import numpy as np


class PKSampler:
    """Yields index arrays into a combined (user + shop) embedding table.

    `keys`: match_key per row; `is_user`: bool per row. Only keys that have both a user and a shop
    row are eligible (an anchor needs a positive). Each epoch visits every eligible key once.
    """

    def __init__(self, keys: np.ndarray, is_user: np.ndarray, P: int, K: int, seed: int = 0) -> None:
        self.P, self.K = P, K
        self.rng = np.random.default_rng(seed)
        self.user_idx: dict[str, np.ndarray] = {}
        self.shop_idx: dict[str, np.ndarray] = {}
        order = np.argsort(keys, kind="stable")
        for k, grp in _group(keys[order], order):
            u, s = grp[is_user[grp]], grp[~is_user[grp]]
            if len(u) and len(s):
                self.user_idx[k], self.shop_idx[k] = u, s
        self.keys = np.array(sorted(self.user_idx))

    def __len__(self) -> int:
        return int(np.ceil(len(self.keys) / self.P))

    def __iter__(self):
        perm = self.rng.permutation(len(self.keys))
        for b in range(len(self)):
            batch = []
            for k in self.keys[perm[b * self.P:(b + 1) * self.P]]:
                n_user = max(1, self.K // 2)
                n_shop = max(1, self.K - n_user)
                batch.extend(self.rng.choice(self.user_idx[k], n_user, replace=len(self.user_idx[k]) < n_user))
                batch.extend(self.rng.choice(self.shop_idx[k], n_shop, replace=len(self.shop_idx[k]) < n_shop))
            yield np.array(batch)


def _group(sorted_keys: np.ndarray, sorted_idx: np.ndarray):
    start = 0
    for i in range(1, len(sorted_keys) + 1):
        if i == len(sorted_keys) or sorted_keys[i] != sorted_keys[start]:
            yield sorted_keys[start], sorted_idx[start:i]
            start = i
