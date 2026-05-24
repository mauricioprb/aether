"""Network-free tests for Etapa 2: graph builder PBC + RunLogger."""

from __future__ import annotations

import json

import matplotlib
import pytest
from ase.build import add_adsorbate, fcc111

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.graph_builder import atoms_to_graph  # noqa: E402
from training.run_logger import RunLogger  # noqa: E402


@pytest.fixture
def slab_with_h():
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=8.0)
    add_adsorbate(slab, "H", height=1.5, position="ontop")
    return slab


def test_graph_builder_basic(slab_with_h):
    g = atoms_to_graph(slab_with_h, y=0.12, cutoff=6.0, sid="x")
    assert g.num_nodes == len(slab_with_h)
    assert g.edge_index.shape[0] == 2
    assert float(g.edge_weight.max()) <= 6.0001
    assert float(g.y.item()) == pytest.approx(0.12)
    assert set(g.tags.tolist()) <= {0, 1, 2}
    assert (g.tags == 2).sum().item() == 1


def test_graph_builder_no_vacuum_edges(slab_with_h):
    g = atoms_to_graph(slab_with_h, y=0.0, cutoff=6.0)
    pos, ei = g.pos.numpy(), g.edge_index.numpy()
    z_span = abs(pos[ei[0], 2] - pos[ei[1], 2]).max()
    cell_z = float(g.cell[0, 2, 2])
    assert z_span < cell_z


def test_run_logger(tmp_path):
    cfg = {"model": "dummy", "lr": 0.001}
    with RunLogger(name="unit", config=cfg, results_dir=tmp_path) as run:
        run.log_metrics({"r2_test": 0.9})
        run.log_predictions([1.0, 2.0], [1.1, 1.9], "test")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        run.log_figure(fig, "f.png")
        run_dir = run.run_dir

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["r2_test"] == 0.9
    assert "elapsed_sec" in metrics
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert (run_dir / "figures" / "f.png").exists()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert any(e["name"] == "unit" for e in summary)
