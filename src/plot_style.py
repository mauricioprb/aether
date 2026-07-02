"""Estilo único das figuras para a dissertação (ABNT, PT-BR, Times New Roman).

Todas as figuras do projeto passam por ``apply_abnt_style()`` e por ``save_fig``,
de modo que fonte, tamanhos e rótulos ficam consistentes e em português. A
fonte segue uma cadeia de fallback: usa Times New Roman real quando instalada
(máquina do autor / TeX Live), senão uma equivalente metric-compatible
(Liberation Serif, Nimbus Roman) e, por fim, STIX/DejaVu Serif, ambas de
aparência Times. O texto matemático usa o conjunto ``stix`` para casar com o
serif do corpo.

Uso:
    from plot_style import apply_abnt_style, save_fig, L
    apply_abnt_style()
    ...
    ax.set_xlabel(L["dg_dft"])
    save_fig(fig, "fig4_parity")   # grava PNG (300 dpi) + PDF vetorial
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Ordem de preferência: Times real -> equivalentes -> Times-like sempre presente.
SERIF_CHAIN = [
    "Times New Roman", "Liberation Serif", "Nimbus Roman",
    "Nimbus Roman No9 L", "STIX", "STIX Two Text", "DejaVu Serif",
]

# Rótulos centralizados em português (fonte única da verdade).
L = {
    "dg_dft": r"$\Delta G_{\mathrm{H}}$ DFT (eV)",
    "dg_pred": r"$\Delta G_{\mathrm{H}}$ predito (eV)",
    "dg": r"$\Delta G_{\mathrm{H}}$ (eV)",
    "residuo": "Resíduo (eV) = predito $-$ DFT",
    "contagem": "Contagem",
    "frac_teste": "Fração das amostras de teste",
    "limiar_erro": "Limiar de $|$erro$|$ (eV)",
    "acuracia_quimica": "acurácia química (43 meV)",
    "mae_meV": "MAE (meV)",
    "importancia_shap": "Importância média $|$SHAP$|$",
    "treino": "treino",
    "teste": "teste",
}


def _resolve_serif() -> str:
    """First font in SERIF_CHAIN actually available; logs the choice."""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in SERIF_CHAIN:
        if name in available:
            if name != "Times New Roman":
                logger.info("Times New Roman ausente; usando '%s' (aparência Times)", name)
            return name
    logger.warning("nenhuma fonte serif da cadeia encontrada; usando serif padrão")
    return "serif"


def apply_abnt_style(base_size: int = 11) -> None:
    """Configura rcParams globais para figuras ABNT em português."""
    serif = _resolve_serif()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif, *SERIF_CHAIN],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size + 2,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig: plt.Figure, name: str, fig_dir: str | Path,
             formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Grava a figura em cada formato. PDF (vetorial) é o preferido pelo LaTeX."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(fig_dir / f"{name}.{ext}")
    logger.info("figura salva: %s.{%s}", name, ",".join(formats))
