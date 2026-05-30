"""Aggregate multi-seed runs into mean ± std tables.

Used after multi-seed training to characterise model variance. Groups runs by
``name`` prefix (e.g. ``schnet_v2_seed42``, ``schnet_v2_seed1``, ... grouped
under ``schnet_v2``) and computes mean/std/min/max for every numeric metric.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS = Path("results")

# Metrics summarised in the markdown table (in order).
METRICS_ORDER = [
    "r2_test", "mae_test", "rmse_test", "mdae_test", "max_err_test",
    "pearson_r_test", "spearman_rho_test", "mae_meV_test", "frac_chem_acc_test",
]

# How each metric is formatted in the markdown table.
def _fmt(metric: str, mean: float, std: float) -> str:
    if metric == "frac_chem_acc_test":
        return f"{100 * mean:.1f} ± {100 * std:.1f}"
    if metric == "mae_meV_test":
        return f"{mean:.0f} ± {std:.0f}"
    return f"{mean:.4f} ± {std:.4f}"


@dataclass
class GroupResult:
    group: str
    display: str
    n_runs: int
    seeds: list[int]
    means: dict[str, float]
    stds: dict[str, float]
    mins: dict[str, float]
    maxs: dict[str, float]
    medians: dict[str, float]
    entries: list[dict]


def _group_key(name: str, group_pattern: str) -> str | None:
    """Return canonical group name when ``name`` matches ``group_pattern``."""
    # Pattern supports trailing _seed{N} or _seed_{N} suffix.
    pattern = re.compile(rf"^{re.escape(group_pattern)}_seed(\d+)$")
    return group_pattern if pattern.match(name) else None


def _extract_seed(name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", name)
    return int(m.group(1)) if m else None


def aggregate_groups(
    groups: list[tuple[str, str]],
    summary_path: Path = RESULTS / "summary.json",
) -> list[GroupResult]:
    """``groups`` is a list of (group_pattern, display_name) tuples."""
    summary = json.loads(summary_path.read_text())
    bucket: dict[str, list[dict]] = defaultdict(list)
    display_map: dict[str, str] = {}
    for pattern, display in groups:
        display_map[pattern] = display
        for entry in summary:
            name = entry.get("name", "")
            if _group_key(name, pattern):
                bucket[pattern].append(entry)

    results: list[GroupResult] = []
    for pattern, display in groups:
        runs = bucket.get(pattern, [])
        if not runs:
            continue
        seeds = sorted(_extract_seed(r["name"]) for r in runs if _extract_seed(r["name"]) is not None)
        # Collect numeric metrics across runs.
        numeric_keys = set()
        for r in runs:
            for k, v in r.items():
                if isinstance(v, (int, float)) and v is not None:
                    numeric_keys.add(k)
        means, stds, mins, maxs, medians = {}, {}, {}, {}, {}
        for k in numeric_keys:
            vals = np.array([r[k] for r in runs if isinstance(r.get(k), (int, float))], dtype=float)
            if vals.size == 0:
                continue
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            mins[k] = float(np.min(vals))
            maxs[k] = float(np.max(vals))
            medians[k] = float(np.median(vals))
        results.append(GroupResult(
            group=pattern,
            display=display,
            n_runs=len(runs),
            seeds=seeds,
            means=means,
            stds=stds,
            mins=mins,
            maxs=maxs,
            medians=medians,
            entries=runs,
        ))
    return results


def markdown_table(results: list[GroupResult]) -> str:
    header_cells = ["Modelo", "n", "seeds"]
    for m in METRICS_ORDER:
        header_cells.append(_METRIC_LABEL.get(m, m))
    header = "| " + " | ".join(header_cells) + " |\n"
    sep = "|" + "|".join(["---"] * len(header_cells)) + "|\n"
    body = []
    for r in results:
        cells = [r.display, str(r.n_runs), ",".join(map(str, r.seeds))]
        for m in METRICS_ORDER:
            if m in r.means:
                cells.append(_fmt(m, r.means[m], r.stds[m]))
            else:
                cells.append("-")
        body.append("| " + " | ".join(cells) + " |")
    return header + sep + "\n".join(body) + "\n"


def per_seed_table(results: list[GroupResult]) -> str:
    """One row per (group, seed) — useful for inspecting variance."""
    cols = ["Modelo", "seed", "R²", "MAE (eV)", "MAE (meV)", "RMSE (eV)", "% < 43 meV", "epoch_best"]
    header = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    rows = []
    for r in results:
        for entry in sorted(r.entries, key=lambda e: _extract_seed(e["name"]) or -1):
            seed = _extract_seed(entry["name"])
            r2 = entry.get("r2_test")
            mae = entry.get("mae_test")
            mae_meV = entry.get("mae_meV_test", (1000 * mae) if mae else None)
            rmse = entry.get("rmse_test")
            frac = entry.get("frac_chem_acc_test")
            ckpt = entry.get("best_ckpt", "")
            ep = ckpt.split("epoch=")[1].split("-")[0] if "epoch=" in ckpt else "-"
            rows.append(
                f"| {r.display} | {seed} | {r2:.4f} | {mae:.4f} | {1000*mae:.0f} | "
                f"{rmse:.4f} | {100*frac:.1f} | {ep} |"
            )
    return header + sep + "\n".join(rows) + "\n"


_METRIC_LABEL = {
    "r2_test": "R²",
    "mae_test": "MAE (eV)",
    "rmse_test": "RMSE (eV)",
    "mdae_test": "MDAE (eV)",
    "max_err_test": "max err (eV)",
    "pearson_r_test": "Pearson r",
    "spearman_rho_test": "Spearman ρ",
    "mae_meV_test": "MAE (meV)",
    "frac_chem_acc_test": "% < 43 meV",
}


# Default groupings used by the dissertation pipeline.
DEFAULT_GROUPS: list[tuple[str, str]] = [
    ("schnet_v2", "SchNet (do zero)"),
    ("mace_ft_stageA_v2", "MACE Stage A (GNN)"),
]
