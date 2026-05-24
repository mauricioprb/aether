from __future__ import annotations

import logging
from pathlib import Path

from ase.io import Trajectory
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from data.graph_builder import DEFAULT_CUTOFF, atoms_to_graph

logger = logging.getLogger(__name__)

TRAJ_PATH = Path("data/processed/her_dataset.traj")


class HERDataset(InMemoryDataset):
    def __init__(self, root: str = "data", traj_path: Path = TRAJ_PATH,
                 cutoff: float = DEFAULT_CUTOFF, transform=None):
        self.traj_path = Path(traj_path)
        self.cutoff = cutoff
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return [f"her_pyg_c{self.cutoff:g}.pt"]

    def process(self) -> None:
        data_list = []
        for atoms in tqdm(Trajectory(str(self.traj_path)), desc="building graphs"):
            data_list.append(atoms_to_graph(
                atoms,
                y=float(atoms.info["delta_G_H"]),
                cutoff=self.cutoff,
                sid=atoms.info.get("id"),
            ))
        logger.info("built %d graphs", len(data_list))
        self.save(data_list, self.processed_paths[0])
