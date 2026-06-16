"""Paradigma 1 — Métodos Exactos (Branch & Cut) con Gurobi.

Referencia de costo óptimo determinista. Resuelve el CVRP sobre el **tiempo de viaje
nominal** ``τ_ij = d_ij + retraso_de_congestión(t*)`` (objetivo de tiempo de viaje de
SVRPBench, no solo distancia) con la formulación de flujo **no dirigida** de dos
índices y **branch-and-cut**: desigualdades de **capacidad redondeada / eliminación de
subtours (RCI/DFJ)** separadas como *lazy constraints* sobre soluciones **enteras**
(``MIPSOL``) — correcto y exacto. Gurobi aporta además sus propios cortes.

    min  Σ_{e={i,j}} τ_e y_e
    s.a. Σ_{e∋h} y_e = 2 ∀ cliente h ;  Σ_j y_{0j} = 2K
         Σ_{e⊆S} y_e ≤ |S| − k(S)   ∀ S ⊆ clientes   [lazy],   k(S)=⌈demanda(S)/cap⌉

La formulación no dirigida (~n²/2 variables) cabe en la licencia gratuita restringida
de Gurobi hasta n≈63; con licencia académica escala a n grandes. La ruta a priori se
evalúa con el evaluador estocástico compartido; las ventanas/retrasos entran como
recurso de 2ª etapa (ver ``exact-bc-tw`` si se quiere un baseline que respete ventanas
en el MIP).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from vrp_bench.core import Instance, Solution, Solver
from .. import stochastic


def _ekey(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _components(adj: Dict[int, List[int]], nodes: List[int]) -> List[List[int]]:
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


def validate_cvrp_routes(routes, demands, cap, customers, *, gap=None, tol=1e-6) -> None:
    """Defensa en profundidad: verifica invariantes de la solución CVRP exacta y lanza
    ``ValueError`` si se violan (cada cliente servido una vez; ninguna ruta excede la
    capacidad; gap no negativo). Un baseline exacto que "mienta" contaminaría todo."""
    dem = np.asarray(demands, dtype=float)
    served = sorted(int(c) for r in routes for c in r)
    expected = sorted(int(c) for c in customers)
    if served != expected:
        raise ValueError(f"validación CVRP: clientes servidos {served} != esperados {expected}")
    for r in routes:
        d = float(dem[list(r)].sum()) if r else 0.0
        if d > cap * (1.0 + tol):
            raise ValueError(f"validación CVRP: ruta {list(r)} demanda={d:.1f} > cap={cap:.1f}")
    if gap is not None and np.isfinite(gap) and gap < -tol:
        raise ValueError(f"validación CVRP: gap negativo {gap}")


def violated_rci(customers, demands, cap, pair_val, *, thr=1e-6, eps=0.5):
    """Separa desigualdades RCI/DFJ violadas por componentes conexas del grafo soporte."""
    adj: Dict[int, List[int]] = {c: [] for c in customers}
    m = len(customers)
    for a in range(m):
        i = customers[a]
        for b in range(a + 1, m):
            j = customers[b]
            if pair_val(i, j) > thr:
                adj[i].append(j); adj[j].append(i)
    out = []
    for S in _components(adj, customers):
        if len(S) < 2:
            continue
        dem = float(sum(demands[i] for i in S))
        k = max(1, math.ceil(dem / cap)) if cap > 0 else 1
        within = 0.0
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                within += pair_val(S[a], S[b])
        if within > len(S) - k + eps:
            out.append((S, k))
    return out


class ExactBranchCut(Solver):
    """CVRP exacto por Branch & Cut (Gurobi) + evaluación estocástica SVRPBench."""

    name = "exact-bc"

    def __init__(self, *, time_limit: float = 120.0, mip_gap: float = 0.0,
                 default_realizations: int = 200, alpha: float = 0.95,
                 late_penalty: float = 1.0, accident_scale: float = 1.0,
                 threads: int = 1, verbose: bool = False):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale
        self.threads = threads
        self.verbose = verbose

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        import gurobipy as gp
        from gurobipy import GRB

        depot = int(instance.metadata.get("depot_index", 0))
        n = instance.num_nodes
        customers = [i for i in range(n) if i != depot]
        demands = np.asarray(instance.demands, dtype=float)
        cap = float(np.asarray(instance.vehicle_capacities, dtype=float).ravel()[0])
        dist = stochastic.euclidean_int_matrix(instance.locations)
        t_star = stochastic.representative_time(instance, depot)
        tau = stochastic.nominal_time_matrix(dist, t_star)

        m = gp.Model("exact_bc_cvrp")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.Threads = self.threads
        m.Params.LazyConstraints = 1

        y: Dict[Tuple[int, int], "gp.Var"] = {}
        for a in range(n):
            for b in range(a + 1, n):
                ub = 2 if (a == depot or b == depot) else 1
                vt = GRB.INTEGER if ub == 2 else GRB.BINARY
                y[a, b] = m.addVar(vtype=vt, lb=0, ub=ub, obj=float(tau[a, b]), name=f"y_{a}_{b}")
        m.ModelSense = GRB.MINIMIZE

        for h in customers:
            m.addConstr(gp.quicksum(y[_ekey(h, k)] for k in range(n) if k != h) == 2)

        k_min = max(1, math.ceil(float(demands.sum()) / cap)) if cap > 0 else 1
        K = m.addVar(vtype=GRB.INTEGER, lb=k_min, ub=len(customers), name="K")
        m.addConstr(gp.quicksum(y[_ekey(depot, j)] for j in customers) == 2 * K)

        conv_log: List[Tuple[float, float, float]] = []
        state = {"bst": None, "bnd": None}

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(list(y.values()))
                yval = dict(zip(y.keys(), vals))
                for S, k in violated_rci(customers, demands, cap,
                                         lambda i, j: yval[_ekey(i, j)], eps=0.5):
                    expr = gp.quicksum(y[_ekey(S[a], S[b])]
                                       for a in range(len(S)) for b in range(a + 1, len(S)))
                    model.cbLazy(expr <= len(S) - k)
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
                f"Gurobi falló en n={n} (¿licencia/tamaño?): {e}. La formulación no "
                "dirigida cabe hasta n≈63 en la licencia restringida; usa licencia "
                "académica para n mayores.") from e
        solve_time = time.time() - t0

        routes = self._extract_routes(y, m, depot, n)
        det_cost = float(m.ObjVal) if m.SolCount > 0 else float("nan")
        gap = float(m.MIPGap) if m.SolCount > 0 else float("nan")
        if m.SolCount > 0:
            validate_cvrp_routes(routes, demands, cap, customers, gap=gap)

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)

        extras = score.as_extras()
        extras.update({"det_cost": det_cost, "gap": gap, "mip_status": int(m.Status),
                       "bc_nodes": float(m.NodeCount), "convergence_log": conv_log,
                       "n_routes": len(routes), "realizations": R})
        return Solution(routes=routes, total_cost=score.expected_cost, runtime=solve_time,
                        feasibility=score.feasibility, cvr=score.cvr,
                        waiting_time=score.waiting_time, robustness=score.robustness,
                        extras=extras)

    @staticmethod
    def _extract_routes(y, model, depot: int, n: int) -> List[List[int]]:
        if model.SolCount == 0:
            return []
        rem: Dict[Tuple[int, int], int] = {e: int(round(var.X)) for e, var in y.items() if round(var.X) > 0}
        routes: List[List[int]] = []
        guard_outer = 0
        while guard_outer <= n + 1:
            guard_outer += 1
            stub = None
            for (a, b), c in rem.items():
                if c > 0 and (a == depot or b == depot):
                    stub = (a, b, b if a == depot else a)
                    break
            if stub is None:
                break
            rem[_ekey(stub[0], stub[1])] -= 1
            cur = stub[2]
            route = [cur]
            guard = 0
            while guard <= n + 1:
                guard += 1
                nxt = ekey = None
                for (a, b), c in rem.items():
                    if c > 0 and (a == cur or b == cur):
                        nxt = b if a == cur else a
                        ekey = (a, b)
                        break
                if nxt is None:
                    break
                rem[ekey] -= 1
                if nxt == depot:
                    break
                route.append(nxt)
                cur = nxt
            routes.append(route)
        return routes
