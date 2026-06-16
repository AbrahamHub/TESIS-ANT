"""Paradigma 2 — Metaheurísticas (ACO y Tabu Search).

Envuelve las implementaciones **oficiales** de SVRPBench (``aco_solver.ACOSolver`` y
``tabu_search_solver.TabuSearchSolver``, ambas con arranque NN+2opt — el "Ant System
genuino" con regla de transición τ^α·η^β, evaporación y depósito Q/L, y la búsqueda
tabú con lista tabú y reparación) y **re-puntúa sus rutas con el evaluador estocástico
compartido** (CRN), de modo que sus métricas son directamente comparables con el resto
de paradigmas (mismo ξ, mismo recurso/CVaR).

Reproducibilidad: los solvers heredados muestrean con los RNG globales ``random`` y
``np.random`` durante la construcción/búsqueda, así que se **siembra e instancia el
solver DESPUÉS de sembrar**, dentro del bucle, dejando el resultado determinado por
``(instancia, semilla)``. El CRN de puntuación usa la semilla de la **instancia** (no
la del algoritmo), así todas las rutas se evalúan sobre los mismos escenarios ξ.

Agregación best-of-K (multistart): se corren ``n_seeds`` semillas y se reporta TODO
(métricas + ruta) de la **misma** corrida elegida (menor E[c+Q]); ``seed_std_*``
documenta la variabilidad entre semillas.
"""
from __future__ import annotations

import random
import time
from statistics import pstdev
from typing import List, Type

import numpy as np

from vrp_bench.core import Instance, Solution, Solver
from .. import stochastic

# Solvers oficiales (path plano provisto por bootstrap).
from aco_solver import ACOSolver as _ACO            # noqa: E402
from tabu_search_solver import TabuSearchSolver as _Tabu  # noqa: E402


class _LegacyMetaheuristic(Solver):
    legacy_cls: Type = None

    def __init__(self, *, default_realizations: int = 200, alpha: float = 0.95,
                 late_penalty: float = 1.0, accident_scale: float = 1.0,
                 n_seeds: int = 5):
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
        runs = []
        for k in range(self.n_seeds):
            s = (base_seed * 7919 + k * 104729) & 0x7FFFFFFF
            random.seed(s); np.random.seed(s)
            legacy = self.legacy_cls(legacy_dict)
            self._apply_params(legacy)
            t0 = time.time()
            res = legacy.solve_instance(0, num_realizations=1)
            rt = time.time() - t0
            routes = self._strip(res.get("routes", []), depot)
            score = stochastic.score_routes(
                instance, routes, num_realizations=R, seed=base_seed, alpha=self.alpha,
                late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)
            runs.append((score, routes, rt, float(res.get("total_cost", float("nan")))))

        best = min(runs, key=lambda r: r[0].expected_total)
        best_score, best_routes, _, best_legacy = best
        ec = [r[0].expected_cost for r in runs]
        fe = [r[0].feasibility for r in runs]

        extras = best_score.as_extras()
        extras.update({
            "seed_std_cost": pstdev(ec) if len(ec) > 1 else 0.0,
            "seed_std_feasibility": pstdev(fe) if len(fe) > 1 else 0.0,
            "n_seeds": self.n_seeds, "n_routes": len(best_routes),
            "legacy_cost": best_legacy, "realizations": R,
            "aggregation": "best-of-K (mín E[c+Q])",
        })
        return Solution(routes=best_routes, total_cost=best_score.expected_cost,
                        runtime=sum(r[2] for r in runs), feasibility=best_score.feasibility,
                        cvr=best_score.cvr, waiting_time=best_score.waiting_time,
                        robustness=best_score.robustness, extras=extras)


class ACO(_LegacyMetaheuristic):
    """Optimización por Colonia de Hormigas (Ant System oficial) re-puntuada con CRN."""
    name = "aco"
    legacy_cls = _ACO

    def __init__(self, *, num_ants: int = 10, max_iterations: int = 20, **kw):
        super().__init__(**kw)
        self.num_ants = num_ants
        self.max_iterations = max_iterations

    def _apply_params(self, legacy) -> None:
        legacy.num_ants = self.num_ants
        legacy.max_iterations = self.max_iterations


class Tabu(_LegacyMetaheuristic):
    """Búsqueda Tabú (oficial) re-puntuada con CRN."""
    name = "tabu"
    legacy_cls = _Tabu

    def __init__(self, *, max_iterations: int = 30, tabu_tenure: int = 5, **kw):
        super().__init__(**kw)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure

    def _apply_params(self, legacy) -> None:
        legacy.max_iterations_base = self.max_iterations
        legacy.tabu_tenure_base = self.tabu_tenure
