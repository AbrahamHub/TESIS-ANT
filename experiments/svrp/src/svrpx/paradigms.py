"""Registro de paradigmas: mapea cada solver a su paradigma del anteproyecto y
define la carpeta de salida correspondiente, para que ``results/`` y ``figures/``
queden organizados por tipo de método.

  results/01_exact/         figures/01_exact/         (Métodos Exactos)
  results/02_metaheuristic/ figures/02_metaheuristic/ (ACO / Tabu)
  results/03_nco_supervised/ ...                       (NCO RL supervisado)
  results/04_nco_pomo_am/   ...                        (NCO determinista POMO/AM)
  results/05_ehbg_facs/     ...                        (propuesta EHBG-FACS)
  results/cross/            ...                        (comparación entre paradigmas)
"""
from __future__ import annotations

from typing import List, Tuple

# solver -> (número de paradigma, slug de carpeta)
PARADIGM_OF = {
    "exact-bc": (1, "exact"),
    "exact-bc-tw": (1, "exact"),
    "aco": (2, "metaheuristic"),
    "tabu": (2, "metaheuristic"),
    "nn2opt": (2, "metaheuristic"),
    "nco-sl": (3, "nco_supervised"),
}

PARADIGM_LABEL = {
    1: "Métodos Exactos (Branch & Cut)",
    2: "Metaheurísticas (ACO / Tabu)",
    3: "NCO (RL supervisado)",
    4: "NCO determinista (POMO / AM)",
    5: "EHBG-FACS (propuesta)",
}


def paradigm_dir(solver_names: List[str]) -> str:
    """Slug de carpeta para un conjunto de solvers. Si todos son del mismo
    paradigma -> ``NN_slug``; si se mezclan paradigmas -> ``cross``."""
    seen = {PARADIGM_OF.get(s, (9, "other")) for s in solver_names}
    if len(seen) == 1:
        num, slug = next(iter(seen))
        return f"{num:02d}_{slug}"
    return "cross"


def paradigm_info(solver_name: str) -> Tuple[int, str, str]:
    num, slug = PARADIGM_OF.get(solver_name, (9, "other"))
    return num, slug, PARADIGM_LABEL.get(num, "Otro")
