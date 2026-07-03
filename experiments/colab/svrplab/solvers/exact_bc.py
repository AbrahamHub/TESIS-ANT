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


def greedy_nn_routes(tau: np.ndarray, demands: np.ndarray, cap: float,
                     customers: List[int], depot: int = 0) -> List[List[int]]:
    """Constructivo vecino-más-cercano con corte por capacidad. Doble uso:
    (a) **arranque MIP** (incumbente inicial del B&C → incumbentes tempranos y
    nunca ``SolCount=0`` dentro del límite de tiempo); (b) **fallback** para
    garantizar que el solver siempre devuelve una ruta a priori válida."""
    pend = set(customers)
    routes: List[List[int]] = []
    while pend:
        cur, rem, route = depot, float(cap), []
        while True:
            cands = [c for c in pend if demands[c] <= rem + 1e-9]
            if not cands:
                break
            nxt = min(cands, key=lambda c: tau[cur, c])
            route.append(nxt); pend.discard(nxt)
            rem -= float(demands[nxt]); cur = nxt
        if not route:      # ningún cliente cabe (no debería pasar: cap >= dem_max)
            nxt = min(pend, key=lambda c: tau[depot, c])
            route = [nxt]; pend.discard(nxt)
        routes.append(route)
    return routes


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

        # Arranque MIP: incumbente golosa NN-capacidad → el B&C parte de una solución
        # factible (incumbentes tempranos; con límite de tiempo nunca sale vacío).
        start_routes = greedy_nn_routes(tau, demands, cap, customers, depot)
        start_count: Dict[Tuple[int, int], int] = {}
        for r in start_routes:
            path = [depot] + list(r) + [depot]
            for a, b in zip(path[:-1], path[1:]):
                e = _ekey(a, b)
                start_count[e] = start_count.get(e, 0) + 1
        for e, var in y.items():
            var.Start = start_count.get(e, 0)
        K.Start = len(start_routes)

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

        fallback = m.SolCount == 0
        if fallback:
            # Nunca devolver vacío: la ruta a priori golosa se usa como recurso
            # (queda declarada en extras; det_cost/gap = NaN porque NO es incumbente).
            routes = start_routes
            det_cost = float("nan")
            gap = float("nan")
        else:
            routes = self._extract_routes(y, m, depot, n)
            det_cost = float(m.ObjVal)
            gap = float(m.MIPGap)
            validate_cvrp_routes(routes, demands, cap, customers, gap=gap)

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)

        extras = score.as_extras()
        extras.update({"det_cost": det_cost, "gap": gap, "mip_status": int(m.Status),
                       "bc_nodes": float(m.NodeCount), "convergence_log": conv_log,
                       "n_routes": len(routes), "realizations": R,
                       "fallback": fallback})
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


class ExactBranchCutTW(Solver):
    """CVRPTW exacto por Branch & Cut dirigido (MTZ + **ventanas soft**) + evaluación CRN.

    Complementa a ``exact-bc`` (que ignora las ventanas): este solver SÍ busca
    satisfacer las restricciones de ventana en el escenario nominal, penalizando la
    tardanza ``L_j = max(0, t_j − b_j)`` en el objetivo (soft — el caveat de escala de
    SVRPBench hace las ventanas duras casi insatisfacibles desde n≈20). Objetivo y
    propagación MTZ usan el tiempo nominal τ (no solo distancia), de modo que la
    infactibilidad residual bajo ξ es atribuible a la incertidumbre, no a un horario
    optimista.

        min  Σ_{i≠j} τ_ij x_ij + λ_TW Σ_j L_j
        s.a. grado 1 por cliente; K rutas; t_j ≥ t_i + τ_ij − M(1−x_ij); t_j ≥ a_j;
             L_j ≥ t_j − b_j;  RCI de capacidad como lazy (solo enteras, Threads=1).

    Tamaño: ~n² binarias ⇒ la licencia restringida cubre n≤~42; n≥50 requiere
    licencia académica. Arranque MIP + fallback golosos: siempre devuelve ruta.
    """

    name = "exact-bc-tw"

    def __init__(self, *, time_limit: float = 120.0, mip_gap: float = 0.0,
                 default_realizations: int = 200, alpha: float = 0.95,
                 late_penalty: float = 1.0, tw_penalty: float = 1.0,
                 accident_scale: float = 1.0, threads: int = 1, verbose: bool = False):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.tw_penalty = tw_penalty
        self.accident_scale = accident_scale
        self.threads = threads   # 1 = separación perezosa correcta (ver exact-bc)
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
        tw = (np.asarray(instance.time_windows, dtype=float)
              if instance.time_windows is not None
              else np.tile([0.0, 1440.0], (n, 1)))

        T_ub = (n + 1) * (float(tau.max()) + 1440.0)
        bigM = T_ub + float(tau.max())

        m = gp.Model("exact_bc_cvrptw")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.Threads = self.threads
        m.Params.LazyConstraints = 1

        x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
             for i in range(n) for j in range(n) if i != j}
        t = {i: m.addVar(lb=0.0, ub=T_ub, name=f"t_{i}") for i in range(n)}
        L = {h: m.addVar(lb=0.0, name=f"L_{h}") for h in customers}

        m.setObjective(
            gp.quicksum(float(tau[i, j]) * x[i, j] for (i, j) in x)
            + self.tw_penalty * gp.quicksum(L[h] for h in customers),
            GRB.MINIMIZE)

        for h in customers:
            m.addConstr(gp.quicksum(x[h, j] for j in range(n) if j != h) == 1)
            m.addConstr(gp.quicksum(x[i, h] for i in range(n) if i != h) == 1)

        k_min = max(1, math.ceil(float(demands.sum()) / cap)) if cap > 0 else 1
        K = m.addVar(vtype=GRB.INTEGER, lb=k_min, ub=len(customers), name="K")
        m.addConstr(gp.quicksum(x[depot, j] for j in customers) == K)
        m.addConstr(gp.quicksum(x[i, depot] for i in customers) == K)

        m.addConstr(t[depot] == 0.0)
        for j in customers:
            a_j, b_j = float(tw[j, 0]), float(tw[j, 1])
            m.addConstr(t[j] >= a_j)
            m.addConstr(L[j] >= t[j] - b_j)
            for i in range(n):
                if i != j:
                    m.addConstr(t[j] >= t[i] + float(tau[i, j]) - bigM * (1 - x[i, j]))

        # Arranque MIP golosa (incumbente inicial; nunca SolCount=0 dentro del límite).
        start_routes = greedy_nn_routes(tau, demands, cap, customers, depot)
        for var in x.values():
            var.Start = 0
        for r in start_routes:
            path = [depot] + list(r) + [depot]
            for a, b in zip(path[:-1], path[1:]):
                x[a, b].Start = 1
        K.Start = len(start_routes)

        conv_log: List[Tuple[float, float, float]] = []
        state = {"bst": None, "bnd": None}

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(list(x.values()))
                xval = dict(zip(x.keys(), vals))
                for S, k in violated_rci(customers, demands, cap,
                                         lambda i, j: xval[i, j] + xval[j, i], eps=0.5):
                    expr = gp.quicksum(x[i, j] for i in S for j in S if i != j)
                    model.cbLazy(expr <= len(S) - k)
            elif where == GRB.Callback.MIP:
                tt = model.cbGet(GRB.Callback.RUNTIME)
                bst = model.cbGet(GRB.Callback.MIP_OBJBST)
                bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                if state["bst"] != bst or state["bnd"] != bnd:
                    state["bst"], state["bnd"] = bst, bnd
                    conv_log.append((float(tt), float(bst), float(bnd)))

        t0 = time.time()
        try:
            m.optimize(callback)
        except gp.GurobiError as e:
            raise RuntimeError(
                f"Gurobi falló en n={n} (¿licencia/tamaño?): {e}. La formulación "
                "dirigida del CVRPTW (~n² binarias) cabe hasta n≈42 en la licencia "
                "restringida; usa licencia académica (WLS) para n mayores.") from e
        solve_time = time.time() - t0

        fallback = m.SolCount == 0
        if fallback:
            routes = start_routes
            det_cost = mip_obj = nominal_late = gap = float("nan")
        else:
            routes = self._extract_routes(x, m, depot, n)
            validate_cvrp_routes(routes, demands, cap, customers,
                                 gap=float(m.MIPGap))
            mip_obj = float(m.ObjVal)
            det_cost = float(sum(tau[i, j] * x[i, j].X for (i, j) in x))
            nominal_late = float(sum(L[h].X for h in customers))
            gap = float(m.MIPGap)

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)

        extras = score.as_extras()
        extras.update({"det_cost": det_cost, "mip_obj": mip_obj,
                       "nominal_tw_lateness": nominal_late, "gap": gap,
                       "mip_status": int(m.Status), "bc_nodes": float(m.NodeCount),
                       "convergence_log": conv_log, "n_routes": len(routes),
                       "realizations": R, "fallback": fallback})
        return Solution(routes=routes, total_cost=score.expected_cost, runtime=solve_time,
                        feasibility=score.feasibility, cvr=score.cvr,
                        waiting_time=score.waiting_time, robustness=score.robustness,
                        extras=extras)

    @staticmethod
    def _extract_routes(x, model, depot: int, n: int) -> List[List[int]]:
        if model.SolCount == 0:
            return []
        sel = {(i, j) for (i, j), var in x.items() if var.X > 0.5}
        succ: Dict[int, int] = {i: j for (i, j) in sel}
        routes: List[List[int]] = []
        for (i, j) in sel:
            if i == depot:
                route, cur, guard = [], j, 0
                while cur != depot and guard <= n:
                    route.append(int(cur))
                    cur = succ.get(cur, depot)
                    guard += 1
                if route:
                    routes.append(route)
        return routes
