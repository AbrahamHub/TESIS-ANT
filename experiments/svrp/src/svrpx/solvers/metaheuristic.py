"""Implementación 2/5 — Metaheurísticas clásicas (ACO y Tabu Search).

Envuelve los solvers **oficiales** de SVRPBench (``aco_solver.ACOSolver`` y
``tabu_search_solver.TabuSearchSolver``, ambos con arranque NN+2opt) y **re-puntúa
sus rutas con el evaluador estocástico compartido** (``svrpx.stochastic``, CRN), de
modo que sus métricas son directamente comparables con el resto de paradigmas (mismo
ξ, mismo recurso/CVaR) en lugar de usar la puntuación interna heredada.

Las rutas heredadas vienen acotadas con el depósito (``[0, c1, ..., 0]``); se
normalizan a índices de cliente (sin depósito) antes de re-puntuar.
"""
from __future__ import annotations

import time
from typing import List, Type

from .._bootstrap import Instance, Solution, Solver
from vrp_bench.core import registry as _registry
from .. import stochastic


def _override_register(name: str, cls):
    """Registra ``cls`` bajo ``name`` reemplazando una entrada previa.

    ``vrp_bench/__init__.py`` auto-importa ``vrp_bench.solvers`` y registra los
    "aco"/"tabu" oficiales (que puntúan con su evaluador interno). Aquí los
    sustituimos por nuestras versiones, que corren el MISMO solver heredado pero
    re-puntúan con el evaluador CRN compartido."""
    _registry._REGISTRY.pop(name, None)
    cls.name = name
    _registry._REGISTRY[name] = cls
    return cls

# Solvers heredados oficiales (imports planos; requieren vrp_bench en el path,
# garantizado por _bootstrap). Dependen de numpy/sklearn/PIL ya instalados.
from aco_solver import ACOSolver as _ACO            # noqa: E402
from tabu_search_solver import TabuSearchSolver as _Tabu  # noqa: E402


class _LegacyMetaheuristic(Solver):
    """Adaptador: corre un ``VRPSolverBase`` heredado y re-puntúa con CRN."""

    legacy_cls: Type = None  # lo fija la subclase

    def __init__(
        self,
        *,
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        accident_scale: float = 1.0,
    ):
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        depot = int(instance.metadata.get("depot_index", 0))

        legacy = self.legacy_cls(instance.to_legacy_dict())
        t0 = time.time()
        res = legacy.solve_instance(0, num_realizations=1)  # solo necesitamos las rutas
        runtime = time.time() - t0

        routes: List[List[int]] = []
        for r in (res.get("routes", []) or []):
            cust = [int(c) for c in r if int(c) != depot]
            if cust:
                routes.append(cust)

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot,
        )

        extras = score.as_extras()
        extras.update({
            "legacy_cost": float(res.get("total_cost", float("nan"))),
            "n_routes": len(routes),
            "realizations": R,
        })
        return Solution(
            routes=routes,
            total_cost=score.expected_cost,
            runtime=runtime,
            feasibility=score.feasibility,
            cvr=score.cvr,
            waiting_time=score.waiting_time,
            robustness=score.robustness,
            extras=extras,
        )


class ACO(_LegacyMetaheuristic):
    """Optimización por Colonia de Hormigas (oficial) re-puntuada con CRN."""
    legacy_cls = _ACO


class Tabu(_LegacyMetaheuristic):
    """Búsqueda Tabú (oficial) re-puntuada con CRN."""
    legacy_cls = _Tabu


_override_register("aco", ACO)
_override_register("tabu", Tabu)
