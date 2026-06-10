"""Implementación 2/5 — Metaheurísticas clásicas (ACO y Tabu Search).

Envuelve los solvers **oficiales** de SVRPBench (``aco_solver.ACOSolver`` y
``tabu_search_solver.TabuSearchSolver``, ambos con arranque NN+2opt) y **re-puntúa
sus rutas con el evaluador estocástico compartido** (``svrpx.stochastic``, CRN), de
modo que sus métricas son directamente comparables con el resto de paradigmas (mismo
ξ, mismo recurso/CVaR) en lugar de usar la puntuación interna heredada.

Correcciones de la revisión de la implementación 2:

  * **R1 — Reproducibilidad.** Los solvers heredados muestrean con ``random`` y
    ``np.random`` (vía ``sample_travel_time``) durante la construcción/búsqueda, así
    que su ruta varía entre corridas. Aquí se **siembra el RNG** de forma determinista
    por instancia, y se ejecutan **``n_seeds`` semillas** para caracterizar la
    naturaleza estocástica del método: se reportan promedios entre semillas y la
    desviación entre semillas (``seed_std_*`` en ``extras``); la ruta representativa
    (la de menor ``E[c+Q]``) se usa para las figuras. El CRN de puntuación usa la
    semilla de la **instancia** (no la del algoritmo), de modo que todas las rutas se
    evalúan sobre los mismos escenarios ξ.

  * **R2 — Hiperparámetros expuestos.** ``num_ants``/``max_iterations`` (ACO) y
    ``max_iterations``/``tabu_tenure`` (Tabu) son parámetros del wrapper (los oficiales
    venían "optimizados para velocidad": 3 hormigas × 5 iteraciones).

  * **R3** — el número de vehículos/rutas se reporta (``n_routes``) para visibilizar
    que las metaheurísticas alcanzan factibilidad usando más rutas cortas.

Notas (revisión):
  * **R4** — las metaheurísticas optimizan su propio objetivo (costo de una sola
    realización), no E[c]/CVaR/factibilidad; se evalúan como baseline con la ruta que
    producen.
  * **R5** — su construcción juzga factibilidad con un escenario muestreado, no con el
    tiempo nominal de los exactos (ligera diferencia de criterio entre paradigmas).

Las rutas heredadas vienen acotadas con el depósito (``[0, c1, ..., 0]``); se
normalizan a índices de cliente (sin depósito) antes de re-puntuar.
"""
from __future__ import annotations

import random
import time
from statistics import mean, pstdev
from typing import List, Type

import numpy as np

from .._bootstrap import Instance, Solution, Solver
from vrp_bench.core import registry as _registry
from .. import stochastic

from aco_solver import ACOSolver as _ACO            # noqa: E402
from tabu_search_solver import TabuSearchSolver as _Tabu  # noqa: E402


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


class _LegacyMetaheuristic(Solver):
    """Adaptador: corre un ``VRPSolverBase`` heredado (sembrado, multi-semilla) y
    re-puntúa con CRN."""

    legacy_cls: Type = None  # lo fija la subclase

    def __init__(
        self,
        *,
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        accident_scale: float = 1.0,
        n_seeds: int = 3,
    ):
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale
        self.n_seeds = max(1, n_seeds)

    def _apply_params(self, legacy) -> None:
        """Las subclases fijan los hiperparámetros del solver heredado."""

    def _strip(self, raw, depot: int) -> List[List[int]]:
        routes = []
        for r in (raw or []):
            cust = [int(c) for c in r if int(c) != depot]
            if cust:
                routes.append(cust)
        return routes

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        depot = int(instance.metadata.get("depot_index", 0))
        base_seed = int(instance.metadata.get("seed", 0))
        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations

        legacy_dict = instance.to_legacy_dict()
        runs = []  # (score, routes, runtime)
        res = None
        for k in range(self.n_seeds):
            s = (base_seed * 7919 + k * 104729) & 0x7FFFFFFF
            random.seed(s)
            np.random.seed(s)
            # Instanciar DESPUÉS de sembrar: así tanto la construcción del solver como
            # su búsqueda quedan totalmente determinadas por (instancia, semilla),
            # independientes del orden de ejecución entre solvers.
            legacy = self.legacy_cls(legacy_dict)
            self._apply_params(legacy)
            t0 = time.time()
            res = legacy.solve_instance(0, num_realizations=1)  # solo necesitamos las rutas
            rt = time.time() - t0
            routes = self._strip(res.get("routes", []), depot)
            score = stochastic.score_routes(
                instance, routes, num_realizations=R, seed=base_seed, alpha=self.alpha,
                late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot,
            )
            runs.append((score, routes, rt))

        # Ruta representativa = la de menor costo total esperado (mejor de n_seeds).
        best_score, best_routes, _ = min(runs, key=lambda x: x[0].expected_total)
        ec = [r[0].expected_cost for r in runs]
        et = [r[0].expected_total for r in runs]
        fe = [r[0].feasibility for r in runs]
        cv = [r[0].cvr for r in runs]
        cvar = [r[0].cvar for r in runs]
        nveh = [len(r[1]) for r in runs]

        extras = best_score.as_extras()
        extras.update({
            "expected_cost": mean(ec),
            "expected_total": mean(et),
            "cvar": mean(cvar),
            "seed_std_cost": pstdev(ec) if len(ec) > 1 else 0.0,
            "seed_std_feasibility": pstdev(fe) if len(fe) > 1 else 0.0,
            "n_seeds": self.n_seeds,
            "n_routes": int(round(mean(nveh))),
            "best_expected_total": best_score.expected_total,
            "legacy_cost": float(res.get("total_cost", float("nan"))),
            "realizations": R,
        })
        return Solution(
            routes=best_routes,
            total_cost=mean(ec),
            runtime=sum(r[2] for r in runs),
            feasibility=mean(fe),
            cvr=mean(cv),
            waiting_time=best_score.waiting_time,
            robustness=best_score.robustness,
            extras=extras,
        )


class ACO(_LegacyMetaheuristic):
    """Optimización por Colonia de Hormigas (Ant System) re-puntuada con CRN."""
    legacy_cls = _ACO

    def __init__(self, *, num_ants: int = 10, max_iterations: int = 20, **kw):
        super().__init__(**kw)
        self.num_ants = num_ants
        self.max_iterations = max_iterations

    def _apply_params(self, legacy) -> None:
        legacy.num_ants = self.num_ants
        legacy.max_iterations = self.max_iterations


class Tabu(_LegacyMetaheuristic):
    """Búsqueda Tabú re-puntuada con CRN."""
    legacy_cls = _Tabu

    def __init__(self, *, max_iterations: int = 30, tabu_tenure: int = 5, **kw):
        super().__init__(**kw)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure

    def _apply_params(self, legacy) -> None:
        legacy.max_iterations_base = self.max_iterations
        legacy.tabu_tenure_base = self.tabu_tenure


_override_register("aco", ACO)
_override_register("tabu", Tabu)
