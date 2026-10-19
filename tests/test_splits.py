import numpy as np
import pandas as pd

from fashret.config import ListRule
from fashret.data.df2_parser import match_key
from fashret.data.splits import build_lists, split_pair_ids


def test_match_key_combines_pair_and_style():
    assert match_key(12, 3) == "12_3"
    assert match_key(12, 3) != match_key(12, 4)


def test_pair_split_has_no_leak_and_is_deterministic():
    ids = np.arange(1000)
    fit1, val1 = split_pair_ids(ids, 0.1, seed=0)
    fit2, val2 = split_pair_ids(ids, 0.1, seed=0)
    assert not (fit1 & val1)
    assert fit1 | val1 == set(ids.tolist())
    assert len(val1) == 100
    assert (fit1, val1) == (fit2, val2)
    assert split_pair_ids(ids, 0.1, seed=1)[1] != val1


def _items():
    rows = [
        # pair 1: user style1 (query), shop style1 (gallery) -> matched
        dict(item_id="a", image_id="1", source="user", pair_id=1, style=1, cropped=True),
        dict(item_id="b", image_id="2", source="shop", pair_id=1, style=1, cropped=True),
        # pair 2: user style2 but shop only style1 -> query has no match -> excluded
        dict(item_id="c", image_id="3", source="user", pair_id=2, style=2, cropped=True),
        dict(item_id="d", image_id="4", source="shop", pair_id=2, style=1, cropped=True),
        # style 0 items: user -> openset, shop -> not gallery
        dict(item_id="e", image_id="5", source="user", pair_id=3, style=0, cropped=True),
        dict(item_id="f", image_id="6", source="shop", pair_id=3, style=0, cropped=True),
        # not cropped -> dropped everywhere
        dict(item_id="g", image_id="7", source="shop", pair_id=1, style=1, cropped=False),
    ]
    df = pd.DataFrame(rows)
    for c in ("category_id", "scale", "occlusion", "zoom_in", "viewpoint"):
        df[c] = 1
    df["crop_path"] = "x.jpg"
    df["match_key"] = [match_key(p, s) for p, s in zip(df["pair_id"], df["style"])]
    return df


def test_build_lists_follow_protocol():
    lists = build_lists(
        _items(),
        gallery=ListRule(source="shop", style_min=1),
        queries=ListRule(source="user", style_min=1, require_match=True),
        openset=ListRule(source="user", style_eq=0),
    )
    assert set(lists["gallery"].item_id) == {"b", "d"}
    assert set(lists["queries"].item_id) == {"a"}
    assert set(lists["openset"].item_id) == {"e"}
