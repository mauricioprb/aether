"""Network-free tests on a synthetic Cu(111) slab with an adsorbed H."""

from __future__ import annotations

import math

import pytest
from ase.build import add_adsorbate, fcc111

from her_gnn.features import FEATURE_NAMES, compute_features, geometric_mean
from her_gnn.filters import classify_site, passes_filters
from her_gnn.geometry import (
    central_indices,
    coordination_numbers,
    h_surface_min_distance,
    neighbor_indices,
    site_type,
)
from her_gnn.storage import assign_tags


@pytest.fixture
def slab_with_h():
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=8.0)
    add_adsorbate(slab, "H", height=1.5, position="ontop")
    return slab


def test_geometric_mean():
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    assert geometric_mean([3.0, 0.0]) == 0.0
    assert math.isnan(geometric_mean([]))


def test_site_type_mapping():
    assert site_type(1) == "top"
    assert site_type(2) == "bridge"
    assert site_type(3) == "hollow"
    assert site_type(4) == "hollow"


def test_geometry_on_ontop(slab_with_h):
    central = central_indices(slab_with_h)
    assert len(central) == 1  # ontop site
    assert classify_site(slab_with_h)[0] == "top"
    neighbors = neighbor_indices(slab_with_h, central)
    assert len(neighbors) > 0
    assert all(c > 0 for c in coordination_numbers(slab_with_h, central))
    bond = h_surface_min_distance(slab_with_h)
    assert 1.0 < bond < 3.0


def test_filters(slab_with_h):
    ok, reason = passes_filters(slab_with_h, delta_g=0.1, coverage=0.11)
    assert ok and reason is None
    assert not passes_filters(slab_with_h, delta_g=3.0, coverage=0.11)[0]
    assert not passes_filters(slab_with_h, delta_g=0.1, coverage=0.5)[0]


def test_features_complete(slab_with_h):
    feats = compute_features(slab_with_h)
    assert feats is not None
    assert set(feats) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) and not math.isnan(v) for v in feats.values())


def test_tags(slab_with_h):
    tags = assign_tags(slab_with_h)
    assert set(tags.tolist()) <= {0, 1, 2}
    assert (tags == 2).sum() == 1  # single adsorbed H
    assert (tags == 1).sum() > 0   # at least one surface atom
