"""A refatoracao so vale se o caminho por Atoms responder o mesmo que o de id.

Se alguem trocar o calculador de embedding, mexer no pooling ou no z_table, os
numeros divergem aqui antes de divergirem em producao silenciosamente.
"""

import numpy as np
import pytest

from data.mace_dataset import TRAJ_PATH, _load_frames
from screening import (
    embed_atoms,
    load_all_embeddings,
    predict_stagea,
    predict_stagea_atoms,
)

pytestmark = pytest.mark.skipif(
    not TRAJ_PATH.exists(), reason="artefatos ausentes (data/processed/her_dataset.traj)"
)

N = 4


@pytest.fixture(scope="module")
def sample():
    frames = _load_frames(str(TRAJ_PATH))
    emb = load_all_embeddings()
    ids = [i for i in emb if i in frames][:N]
    return ids, [frames[i] for i in ids], emb


def test_stagea_id_path_matches_atoms_path(sample):
    ids, atoms, _ = sample
    assert np.allclose(predict_stagea(ids), predict_stagea_atoms(atoms), atol=1e-5)


def test_embedding_recomputed_matches_precomputed(sample):
    """O ETR foi treinado sobre os .npz; se o embedding calculado na hora nao
    reproduzir aqueles vetores, a predicao de estrutura nova sai de outra
    distribuicao de features."""
    ids, atoms, emb = sample
    recomputed = embed_atoms(atoms)
    stored = np.vstack([emb[i] for i in ids])
    assert np.allclose(recomputed, stored, atol=1e-5)


def test_dataset_structures_pass_the_overlap_guard():
    """O limiar foi calibrado no dataset (menor razao observada 0.669). Se alguem
    apertar MIN_DISTANCE_RATIO acima disso, a guarda passa a rejeitar estrutura
    legitima -- e o /predict recusa exatamente o que deveria aceitar."""
    from geometry import MIN_DISTANCE_RATIO

    assert MIN_DISTANCE_RATIO < 0.669


def test_overlapping_atoms_are_flagged():
    from ase.build import add_adsorbate, fcc111

    from geometry import MIN_DISTANCE_RATIO, min_distance_ratio

    bad = fcc111("Pt", size=(3, 3, 3), a=3.92, vacuum=8.0)
    add_adsorbate(bad, "H", height=0.8, position="ontop")
    assert min_distance_ratio(bad) < MIN_DISTANCE_RATIO

    ok = fcc111("Pt", size=(3, 3, 3), a=3.92, vacuum=8.0)
    add_adsorbate(ok, "H", height=1.0, position="fcc")
    assert min_distance_ratio(ok) > MIN_DISTANCE_RATIO
