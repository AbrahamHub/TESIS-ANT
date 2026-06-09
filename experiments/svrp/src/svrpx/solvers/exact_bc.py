"""Implementación 1/5 — Métodos Exactos (Branch & Cut) con Gurobi.

Paradigma de referencia (cota óptima determinista) del anteproyecto. Resuelve el
CVRP determinista (distancia euclidiana entera, escenario nominal sin retrasos)
con la **formulación de flujo de dos índices NO dirigida** y **branch-and-cut
genuino**: desigualdades de **capacidad redondeada / eliminación de subtours
(RCI/DFJ)** separadas como *lazy constraints* sobre soluciones enteras en un
*callback*.

  min  sum_{e={i,j}} d_e y_e
  s.a. sum_{e ∋ h} y_e = 2                 (grado del cliente h)
       sum_{j} y_{0j} = 2 K                 (K rutas desde el depósito)
       sum_{e ⊆ S} y_e <= |S| - k(S)        para todo S ⊆ clientes      [lazy]
       y_e ∈ {0,1} (clientes),  y_{0j} ∈ {0,1,2} (aristas al depósito)
  con k(S) = max(1, ceil(demanda(S)/cap)). La familia RCI fuerza simultáneamente
  la capacidad y la conectividad (elimina subtours).

Se usa la formulación no dirigida (la mitad de variables que la dirigida) para
que los tres tamaños 10/20/50 entren incluso en la licencia restringida que trae
``gurobipy`` (n=51 -> 1275 aristas < 2000). Es además la forma clásica del
Branch & Cut para ruteo.

La ruta a priori óptima se evalúa luego con el evaluador estocástico compartido
(``svrpx.stochastic``): ventanas de tiempo y retrasos log-normales/Poisson entran
como **recurso de 2ª etapa**, no como restricciones duras del MIP.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from .._bootstrap import Instance, Solution, Solver, register_solver
from .. import stochastic


def _ekey(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _components(adj: Dict[int, List[int]], nodes: List[int]) -> List[List[int]]:
    """Componentes conexas del grafo soporte (BFS)."""
    seen, comps = set(), []
    for s in nodes:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


@register_solver("exact-bc")
class ExactBranchCut(Solver):
    """CVRP exacto por Branch & Cut (Gurobi) + evaluación estocástica SVRPBench."""

    def __init__(
        self,
        *,
        time_limit: float = 120.0,
        mip_gap: float = 0.0,
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        threads: int = 0,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.threads = threads
        self.verbose = verbose

    # ----------------------------------------------------------------- solve
    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        import gurobipy as gp
        from gurobipy import GRB

        depot = int(instance.metadata.get("depot_index", 0))
        n = instance.num_nodes
        customers = [i for i in range(n) if i != depot]
        demands = np.asarray(instance.demands, dtype=float)
        cap = float(np.asarray(instance.vehicle_capacities, dtype=float).ravel()[0])
        dist = stochastic.euclidean_int_matrix(instance.locations)

        def k_of(S) -> int:
            dem = float(sum(demands[i] for i in S))
            return max(1, math.ceil(dem / cap)) if cap > 0 else 1

        m = gp.Model("exact_bc_cvrp")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.Threads = self.threads
        m.Params.LazyConstraints = 1

        # Aristas no dirigidas y_e. Aristas al depósito admiten valor 2.
        y: Dict[Tuple[int, int], "gp.Var"] = {}
        for a in range(n):
            for b in range(a + 1, n):
                ub = 2 if (a == depot or b == depot) else 1
                vt = GRB.INTEGER if ub == 2 else GRB.BINARY
                y[a, b] = m.addVar(vtype=vt, lb=0, ub=ub, obj=float(dist[a, b]),
                                   name=f"y_{a}_{b}")
        m.ModelSense = GRB.MINIMIZE

        def inc(h):  # aristas incidentes a h
            return [y[_ekey(h, k)] for k in range(n) if k != h]

        # Grado de clientes = 2.
        for h in customers:
            m.addConstr(gp.quicksum(inc(h)) == 2)

        # Número de rutas K (variable) = mitad del grado del depósito.
        k_min = max(1, math.ceil(float(demands.sum()) / cap)) if cap > 0 else 1
        K = m.addVar(vtype=GRB.INTEGER, lb=k_min, ub=len(customers), name="K")
        m.addConstr(gp.quicksum(y[_ekey(depot, j)] for j in customers) == 2 * K)

        cust_set = set(customers)
        conv_log: List[Tuple[float, float, float]] = []
        state = {"bst": None, "bnd": None}

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                yval = model.cbGetSolution(y)
                adj: Dict[int, List[int]] = {c: [] for c in customers}
                for (a, b), v in yval.items():
                    if v > 0.5 and a in cust_set and b in cust_set:
                        adj[a].append(b)
                        adj[b].append(a)
                for S in _components(adj, customers):
                    k = k_of(S)
                    within = [(_ekey(i, j)) for idx, i in enumerate(S) for j in S[idx + 1:]]
                    cur = sum(yval[e] for e in within)
                    if cur > len(S) - k + 0.5:
                        model.cbLazy(gp.quicksum(y[e] for e in within) <= len(S) - k)
            elif where == GRB.Callback.MIP:
                t = model.cbGet(GRB.Callback.RUNTIME)
                bst = model.cbGet(GRB.Callback.MIP_OBJBST)
                bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                if state["bst"] != bst or state["bnd"] != bnd:
                    state["bst"], state["bnd"] = bst, bnd
                    conv_log.append((float(t), float(bst), float(bnd)))

        t0 = time.time()
        try:
            m.optimize(callback)
        except gp.GurobiError as e:
            raise RuntimeError(
                f"Gurobi falló en n={n} (¿licencia/tamaño?): {e}. "
                "Con la formulación no dirigida n<=~63 cabe en la licencia restringida; "
                "tamaños mayores requieren licencia académica."
            ) from e
        solve_time = time.time() - t0

        routes = self._extract_routes(y, m, depot, n)
        det_cost = float(m.ObjVal) if m.SolCount > 0 else float("nan")
        gap = float(m.MIPGap) if m.SolCount > 0 else float("nan")

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed,
            alpha=self.alpha, late_penalty=self.late_penalty, depot=depot,
        )

        extras = score.as_extras()
        extras.update({
            "det_cost": det_cost,
            "gap": gap,
            "mip_status": int(m.Status),
            "bc_nodes": float(m.NodeCount),
            "convergence_log": conv_log,
            "n_routes": len(routes),
            "realizations": R,
        })
        return Solution(
            routes=routes,
            total_cost=score.expected_cost,
            runtime=solve_time,
            feasibility=score.feasibility,
            cvr=score.cvr,
            waiting_time=score.waiting_time,
            robustness=score.robustness,
            extras=extras,
        )

    # -------------------------------------------------------------- extract
    @staticmethod
    def _extract_routes(y, model, depot: int, n: int) -> List[List[int]]:
        """Reconstruye rutas del grafo no dirigido consumiendo aristas."""
        if model.SolCount == 0:
            return []
        rem: Dict[Tuple[int, int], int] = {
            e: int(round(var.X)) for e, var in y.items() if round(var.X) > 0
        }

        def pop_incident(u, forbid_close=False):
            for (a, b), c in rem.items():
                if c <= 0:
                    continue
                if a == u or b == u:
                    other = b if a == u else a
                    if forbid_close and other == depot and c == 1 and len(rem) > 1:
                        continue
                    return (a, b), other
            return None, None

        routes: List[List[int]] = []
        guard_outer = 0
        while guard_outer <= n + 1:
            guard_outer += 1
            # arrancar una ruta desde una arista del depósito sin consumir
            stub = None
            for (a, b), c in rem.items():
                if c > 0 and (a == depot or b == depot):
                    stub = (a, b, b if a == depot else a)
                    break
            if stub is None:
                break
            ek = _ekey(stub[0], stub[1])
            rem[ek] -= 1
            cur, prev = stub[2], depot
            route = [cur]
            guard = 0
            while guard <= n + 1:
                guard += 1
                e, nxt = pop_incident(cur)
                if nxt is None:
                    break
                rem[e] -= 1
                if nxt == depot:
                    break
                route.append(nxt)
                prev, cur = cur, nxt
            routes.append(route)
        return routes
