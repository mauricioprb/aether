"""Pluggable pretrained backbones for per-atom descriptor extraction.

Every backbone exposes the same contract, so comparing representations
(MACE-MP-0 vs OC20-pretrained vs future foundation models) under the identical
ETR protocol becomes a loop over ``BACKBONES``:

    backbone = BACKBONES["mace_mp_medium"]()
    desc = backbone.descriptors(atoms)   # (n_atoms, dim)

Pooling to a per-structure vector is backbone-agnostic (``pool_descriptors``):
``[desc(H), mean(desc(central atoms < cutoff))]``, matching the published MACE
embedding pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms

from geometry import adsorbate_indices, central_indices

logger = logging.getLogger(__name__)


class EmbeddingBackbone(ABC):
    """A frozen pretrained model that yields per-atom descriptors."""

    name: str

    @abstractmethod
    def descriptors(self, atoms: Atoms) -> np.ndarray:
        """Per-atom descriptor matrix, shape (n_atoms, dim)."""


def pool_descriptors(desc: np.ndarray, atoms: Atoms,
                     cutoff_neighbors: float = 2.4) -> np.ndarray:
    """[desc(H), mean(desc(central))] - the project-wide pooling scheme."""
    h_idx = adsorbate_indices(atoms)
    central = central_indices(atoms, cutoff_neighbors)
    emb_h = desc[h_idx].mean(axis=0)
    emb_n = desc[central].mean(axis=0) if central else emb_h
    return np.concatenate([emb_h, emb_n]).astype(np.float32)


class MACEMPBackbone(EmbeddingBackbone):
    """MACE-MP-0 invariant (L=0) node descriptors (pretrained on MPtrj)."""

    def __init__(self, model: str = "medium", device: str | None = None):
        import torch
        from mace.calculators import mace_mp

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.calc = mace_mp(model=model, device=device, default_dtype="float32")
        self.name = f"mace_mp_{model}"

    def descriptors(self, atoms: Atoms) -> np.ndarray:
        return np.asarray(
            self.calc.get_descriptors(atoms, invariants_only=True), dtype=np.float64
        )


class FairChemBackbone(EmbeddingBackbone):
    """OC20-pretrained fairchem model as descriptor extractor.

    OC20 models saw adsorbate-on-slab chemistry (unlike MACE-MP-0's bulk
    MPtrj), so their representations are domain-matched to HER adsorption.

    Requires the optional ``fairchem-core`` dependency. Node embeddings are
    captured with a forward hook on the backbone's final interaction block;
    the hook point is checkpoint-dependent, so this adapter validates it at
    construction and fails loudly with instructions instead of guessing.
    # TODO: pin fairchem-core version + checkpoint name after the first
    # validated extraction run on GPU.
    """

    def __init__(self, checkpoint: str = "EquiformerV2-31M-S2EF-OC20-All+MD",
                 device: str | None = None):
        try:
            from fairchem.core import OCPCalculator  # fairchem-core >= 1.x
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "fairchem-core não instalado. `uv add fairchem-core` e rode "
                "de novo (GPU recomendada)."
            ) from exc
        import torch

        cpu = not (device == "cuda" or (device is None and torch.cuda.is_available()))
        self.calc = OCPCalculator(model_name=checkpoint, cpu=cpu)
        self.name = f"fairchem_{checkpoint.split('-')[0].lower()}"
        self._captured: list = []
        self._register_hook()

    def _register_hook(self) -> None:
        model = self.calc.trainer.model
        # fairchem wraps the net in module(s); find the deepest module exposing
        # per-node outputs. EquiformerV2: `norm` before the energy head.
        target = None
        for name, mod in model.named_modules():
            if name.endswith(("norm_final", "norm")) and target is None:
                target = mod
        if target is None:
            raise RuntimeError(
                "não encontrei o ponto de hook para node embeddings neste "
                "checkpoint; inspecione `calc.trainer.model.named_modules()` "
                "e ajuste FairChemBackbone._register_hook"
            )
        target.register_forward_hook(
            lambda _m, _inp, out: self._captured.append(out)
        )

    def descriptors(self, atoms: Atoms) -> np.ndarray:
        self._captured.clear()
        atoms = atoms.copy()
        atoms.calc = self.calc
        atoms.get_potential_energy()  # triggers forward + hook
        if not self._captured:
            raise RuntimeError("forward hook não capturou node embeddings")
        out = self._captured[-1]
        emb = out.embedding if hasattr(out, "embedding") else out
        arr = emb.detach().cpu().numpy()
        if arr.ndim == 3:  # (n_atoms, sphere, dim) -> invariant part
            arr = arr[:, 0, :]
        if arr.shape[0] != len(atoms):
            raise RuntimeError(
                f"embedding rows ({arr.shape[0]}) != n_atoms ({len(atoms)}); "
                "hook point errado para este checkpoint"
            )
        return arr.astype(np.float64)


BACKBONES = {
    "mace_mp_medium": lambda: MACEMPBackbone("medium"),
    "mace_mp_small": lambda: MACEMPBackbone("small"),
    "eqv2_oc20": lambda: FairChemBackbone("EquiformerV2-31M-S2EF-OC20-All+MD"),
}
