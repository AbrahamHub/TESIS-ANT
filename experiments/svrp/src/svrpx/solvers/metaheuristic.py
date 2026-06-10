"""Implementación 2/5 — Metaheurísticas clásicas (ACO y Tabu Search).

Envuelve los solvers **oficiales** de SVRPBench (``aco_solver.ACOSolver`` y
``tabu_search_solver.TabuSearchSolver``, ambos con arranque NN+2opt) y **re-puntúa
sus rutas con el evaluador estocástico compartido** (``svrpx.stochastic``, CRN), de
modo que sus métricas son directamente comparables con el resto de paradigmas (mismo
ξ, mismo recurso/CVaR) en lugar de usar la puntuación interna heredada.

Correcciones de la revisión de la implementación 2:

  * **R1 — Reproducibilidad.** Los solvers heredados muestrean con ``random`` y
    ``np.random`` (vía ``sample_travel_time``) durante la construcción/búsqueda, así
    que su ruta varía entre corridas. Aquí se **siembra el RNG e instancia el solver
    DESPUÉS de sembrar**, dentro del bucle, de modo que el resultado queda totalmente
    determinado por ``(instancia, semilla)`` —independiente del orden de ejecución
    entre solvers—. El CRN de puntuación usa la semilla de la **instancia** (no la del
    algoritmo), de modo que todas las rutas se evalúan sobre los mismos escenarios ξ.

  * **M1 — Agregación coherente (best-of-K).** Se ejecutan ``n_seeds`` semillas y se
    reporta TODO (métricas y ruta) de la **misma** corrida elegida: la de menor
    ``E[c+Q]`` (convención multistart estándar). ``seed_std_cost``/``seed_std_feasibility``
    documentan la variabilidad entre semillas. Así la figura de ruta, las barras y la
    tabla se refieren a la misma solución.

  * **M3** — ``n_seeds=5`` por defecto; para afirmaciones estadísticas formales conviene
    10–30 semillas (tradeoff de cómputo).

  * **M4** — el determinismo asume que los solvers heredados usan los RNG **globales**
    ``random``/``np.random`` (verificado empíricamente); un RNG propio
    (``np.random.default_rng``/``random.Random``) no quedaría controlado.

  * **M5** — los baselines NO están sintonizados (defaults razonables, no Optuna); para
    una comparación final justa conviene tunearlos o declarar esta limitación.

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
from statistics import pstdev
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
        n_seeds: int = 5,
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
        runs = []  # (score, routes, runtime, legacy_cost)
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
            runs.append((score, routes, rt, float(res.get("total_cost", float("nan")))))

        # Convención **best-of-K** (multistart): TODAS las métricas reportadas y la ruta
        # provienen de la MISMA corrida elegida (la de menor E[c+Q]); ``seed_std_*``
        # documenta la variabilidad entre semillas. Así la figura de ruta y las barras
        # se refieren a la misma solución (coherencia interna).
        best = min(runs, key=lambda r: r[0].expected_total)
        best_score, best_routes, _, best_legacy = best
        ec = [r[0].expected_cost for r in runs]   # solo para la dispersión entre semillas
        fe = [r[0].feasibility for r in runs]

        extras = best_score.as_extras()  # expected_cost/expected_total/cvar/... de la mejor
        extras.update({
            "seed_std_cost": pstdev(ec) if len(ec) > 1 else 0.0,
            "seed_std_feasibility": pstdev(fe) if len(fe) > 1 else 0.0,
            "n_seeds": self.n_seeds,
            "n_routes": len(best_routes),
            "legacy_cost": best_legacy,
            "realizations": R,
            "aggregation": "best-of-K (mín E[c+Q])",
        })
        return Solution(
            routes=best_routes,
            total_cost=best_score.expected_cost,
            runtime=sum(r[2] for r in runs),
            feasibility=best_score.feasibility,
            cvr=best_score.cvr,
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
