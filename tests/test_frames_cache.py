"""O cache de frames e o que separa /screen de 2 s e de 40 ms: se ele parar de
acertar, a API volta a reler 90 MB por requisicao sem ninguem notar."""

from data.mace_dataset import TRAJ_PATH, _load_frames


def test_frames_are_cached_by_path():
    _load_frames.cache_clear()
    first = _load_frames(str(TRAJ_PATH))
    second = _load_frames(str(TRAJ_PATH))
    assert second is first, "segunda leitura deveria vir do cache"
    assert _load_frames.cache_info().hits == 1
    assert first, "traj vazio: o dataset nao foi construido"
