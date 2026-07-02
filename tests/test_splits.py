"""Unit tests for the canonical split machinery (no sqlite, synthetic frame)."""

from __future__ import annotations

import pandas as pd
import pytest

from data.splits import carve_val, make_splits


@pytest.fixture
def frame() -> pd.DataFrame:
    # 100 structures over 10 compositions (10 structures each)
    return pd.DataFrame({
        "id": [f"s{i}" for i in range(100)],
        "composition": [f"comp{i // 10}" for i in range(100)],
    })


def test_random_split_disjoint_and_complete(frame):
    s = make_splits(frame, strategy="random", test_size=0.2, seed=42)
    train, test = set(s["train"]), set(s["test"])
    assert not train & test
    assert train | test == set(frame["id"])
    assert len(test) == 20
    assert s["strategy"] == "random"


def test_random_split_deterministic(frame):
    a = make_splits(frame, strategy="random", seed=42)
    b = make_splits(frame, strategy="random", seed=42)
    assert a == b


def test_composition_split_no_group_leakage(frame):
    s = make_splits(frame, strategy="composition", test_size=0.2, seed=42)
    comp = dict(zip(frame["id"], frame["composition"], strict=True))
    train_comps = {comp[i] for i in s["train"]}
    test_comps = {comp[i] for i in s["test"]}
    assert not train_comps & test_comps, "composition appears on both sides"
    assert set(s["train"]) | set(s["test"]) == set(frame["id"])


def test_unknown_strategy_raises(frame):
    with pytest.raises(ValueError, match="unknown split strategy"):
        make_splits(frame, strategy="magic")


def test_carve_val_partition():
    train = [f"s{i}" for i in range(50)]
    rest, val = carve_val(train, seed=42, val_frac=0.1)
    assert len(val) == 5
    assert not set(rest) & set(val)
    assert set(rest) | set(val) == set(train)
    # order of remaining train ids is preserved
    assert rest == [i for i in train if i not in set(val)]


def test_carve_val_deterministic():
    train = [f"s{i}" for i in range(200)]
    assert carve_val(train, seed=1) == carve_val(train, seed=1)
    assert carve_val(train, seed=1) != carve_val(train, seed=2)
