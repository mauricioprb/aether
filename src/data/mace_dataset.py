"""MACE-native dataset: converts ASE Atoms -> PyG Data for fine-tuning.

Each structure produces a ``torch_geometric.data.Data`` object with all the
keys expected by MACE's forward method (edge_index, positions, node_attrs, cell,
etc.) plus custom attributes for pooling masks (h_mask, nb_mask) and target (y).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from ase.io import Trajectory
from mace import data as mace_data
from mace.tools import torch_tools
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from data.splits import three_way_split
from geometry import adsorbate_indices, central_indices

logger = logging.getLogger(__name__)

TRAJ_PATH = Path("data/processed/her_dataset.traj")


def _make_data(
    atoms, z_table: torch.Tensor, r_max: float, heads: list[str],
) -> Data:
    """Convert one ASE Atoms to a PyG Data object in MACE-compatible format."""
    with torch_tools.default_dtype(torch.float32):
        config = mace_data.config_from_atoms(atoms)
        ad = mace_data.AtomicData.from_config(
            config, z_table=z_table, cutoff=r_max, heads=heads,
        )
    # Convert to plain PyG Data (avoids AtomicData __init__ issues during batching)
    data = Data()
    for key in ad.keys:
        data[key] = ad[key]
    data.num_nodes = ad.num_nodes
    return data


def _pool_mask(
    node_feats: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool node_feats over nodes selected by mask, per graph.

    Args:
        node_feats: (total_nodes, D)
        batch: (total_nodes,) graph index per node
        num_graphs: int
        mask: (total_nodes,) bool
    Returns:
        (num_graphs, D)
    """
    from torch_geometric.utils import scatter

    feats_masked = node_feats * mask.unsqueeze(-1).to(node_feats.dtype)
    counts = scatter(mask.to(node_feats.dtype), batch, dim=0, dim_size=num_graphs).clamp(min=1)
    sums = scatter(feats_masked, batch, dim=0, dim_size=num_graphs)
    return sums / counts.unsqueeze(-1)


class MACEDataset(Dataset):
    """Dataset that yields PyG Data objects in MACE format for each HER structure."""

    def __init__(
        self,
        traj_path: str | Path = TRAJ_PATH,
        ids: list[str] | None = None,
        z_table: torch.Tensor | None = None,
        r_max: float = 6.0,
        cutoff_neighbors: float = 2.4,
        heads: list[str] | None = None,
        default_dtype: torch.dtype = torch.float32,
    ):
        self.traj_path = Path(traj_path)
        self.cutoff_neighbors = cutoff_neighbors
        self.heads = heads or ["Default"]
        self.r_max = r_max

        if z_table is None:
            self.z_table = torch.zeros(90, dtype=torch.long)
            self.z_table[1:] = torch.arange(1, 89)
        else:
            self.z_table = z_table

        frames = {a.info["id"]: a for a in Trajectory(str(self.traj_path))}
        all_ids = list(frames.keys())
        if ids is not None:
            all_ids = [i for i in ids if i in frames]

        self.data_list: list[Data] = []
        self.ys: list[float] = []
        self.sids: list[str] = []

        for sid in tqdm(all_ids, desc="MACE dataset"):
            atoms = frames[sid]
            data = _make_data(atoms, self.z_table, self.r_max, self.heads)

            # Attach masks for pooling
            h_idx = adsorbate_indices(atoms)
            central = central_indices(atoms, cutoff_neighbors)

            h_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            if len(h_idx):
                h_mask[torch.tensor(h_idx, dtype=torch.long)] = True
            data.h_mask = h_mask

            nb_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            if len(central):
                nb_mask[torch.tensor(central, dtype=torch.long)] = True
            data.nb_mask = nb_mask

            data.y = torch.tensor([float(atoms.info["delta_G_H"])], dtype=torch.float32)
            data.sid = sid

            self.data_list.append(data)
            self.ys.append(float(atoms.info["delta_G_H"]))
            self.sids.append(sid)

        logger.info("MACE dataset: %d structures", len(self.data_list))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


def three_way_split_mace(
    seed: int = 42, val_frac: float = 0.1,
) -> dict[str, list[str]]:
    """Canonical train/val/test partition - delegates to ``data.splits``.

    Historical note: this used to carve val with a torch RNG while the
    embedding pipeline used numpy, so the two saw different val subsets
    (same test set). Unified on the numpy implementation.
    """
    return three_way_split(seed=seed, val_frac=val_frac)
