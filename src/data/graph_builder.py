from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

from storage import assign_tags

SLAB_PBC = (True, True, False)
DEFAULT_CUTOFF = 6.0


def atoms_to_graph(atoms: Atoms, y: float, cutoff: float = DEFAULT_CUTOFF,
                   sid: str | None = None) -> Data:
    atoms = atoms.copy()
    atoms.pbc = SLAB_PBC
    i, j, d, _S = neighbor_list("ijdS", atoms, cutoff=cutoff)

    edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
    edge_weight = torch.tensor(d, dtype=torch.float)
    tags = torch.tensor(assign_tags(atoms), dtype=torch.long)

    return Data(
        z=torch.tensor(atoms.numbers, dtype=torch.long),
        pos=torch.tensor(atoms.positions, dtype=torch.float),
        edge_index=edge_index,
        edge_weight=edge_weight,
        cell=torch.tensor(np.array(atoms.cell), dtype=torch.float).unsqueeze(0),
        tags=tags,
        y=torch.tensor([y], dtype=torch.float),
        natoms=torch.tensor([len(atoms)], dtype=torch.long),
        sid=sid,
    )
