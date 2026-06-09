"""Pone el paquete oficial ``vrp_bench`` (SVRPBench) en el ``sys.path``.

El repo clonado mezcla dos estilos de import:
  * la API moderna ``vrp_bench.core`` (estilo paquete)  -> requiere el padre
    ``third_party/svrpbench`` en el path;
  * los módulos heredados (``city``, ``common``, ``time_windows_generator``)
    se importan por nombre plano -> requieren ``third_party/svrpbench/vrp_bench``
    en el path.

Este módulo agrega ambos y re-exporta los símbolos oficiales que usa el harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

# experiments/svrp/src/svrpx/_bootstrap.py -> experiments/svrp/
_SVRP_ROOT = Path(__file__).resolve().parents[2]
_REPO = _SVRP_ROOT / "third_party" / "svrpbench"
_PKG = _REPO / "vrp_bench"

for _p in (str(_REPO), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not _PKG.exists():  # pragma: no cover
    raise RuntimeError(
        f"No se encontró el paquete oficial vrp_bench en {_PKG}.\n"
        "Clónalo con: git clone https://github.com/yehias21/svrpbench "
        f"{_REPO}"
    )

# API oficial moderna (vrp_bench.core / evaluation).
from vrp_bench.core import Instance, Solution, Solver  # noqa: E402
from vrp_bench.core.registry import register_solver, get_solver, list_solvers  # noqa: E402
from vrp_bench.evaluation import evaluate  # noqa: E402

SVRP_ROOT = _SVRP_ROOT

__all__ = [
    "Instance", "Solution", "Solver",
    "register_solver", "get_solver", "list_solvers",
    "evaluate", "SVRP_ROOT",
]
