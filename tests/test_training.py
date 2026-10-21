"""Sampler and loss sanity checks on tiny synthetic data."""

import numpy as np
import torch

from fashret.data.samplers import PKSampler
from fashret.engine.losses import infonce_user_to_shop
from fashret.models.projection_head import ProjectionHead


def test_pk_sampler_batches_have_user_and_shop_per_key():
    keys = np.array(["a", "a", "a", "b", "b", "c", "c", "d"])        # d has no shop -> ineligible
    is_user = np.array([1, 0, 0, 1, 0, 1, 0, 1], bool)
    s = PKSampler(keys, is_user, P=2, K=2, seed=0)
    assert set(s.keys) == {"a", "b", "c"}
    batches = list(s)
    assert len(batches) == 2
    for b in batches:
        assert len(b) in (2, 4)
        for k in set(keys[b]):
            rows = b[keys[b] == k]
            assert is_user[rows].any() and (~is_user[rows]).any()


def test_infonce_is_lower_when_positives_are_closer():
    keys = torch.tensor([0, 1, 0, 1])
    is_user = torch.tensor([True, True, False, False])
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.1, 1.0], [1.0, 0.1]]), dim=1)
    assert infonce_user_to_shop(good, keys, is_user, 0.1) < infonce_user_to_shop(bad, keys, is_user, 0.1)
    # two classes, symmetric: loss with perfectly separated positives ~ 0
    assert infonce_user_to_shop(good, keys, is_user, 0.01) < 0.01


def test_projection_head_outputs_unit_vectors():
    h = ProjectionHead(8, 16, 4)
    z = h(torch.randn(5, 8))
    assert z.shape == (5, 4)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)
