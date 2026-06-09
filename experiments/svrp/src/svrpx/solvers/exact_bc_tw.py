"""Baseline exacto CVRPTW (Branch & Cut dirigido + MTZ con ventanas de tiempo).

Complementa a ``exact-bc`` (que resuelve CVRP e **ignora** las ventanas) para
**aislar el efecto de la estocasticidad**: este solver SÍ intenta respetar las
ventanas de tiempo (deterministas, conocidas a priori) en el escenario nominal,
de modo que su factibilidad/recurso bajo ξ es atribuible a la incertidumbre y no
a haber ignorado las ventanas.

Formulación de flujo **dirigida** de dos índices con tiempos tipo **MTZ** y
ventanas *soft* (penalización de tardanza en el objetivo, siempre factible):

  min  sum_{i!=j} d_ij x_ij  +  λ · sum_i L_i
  s.a. sum_j x_ij = 1,  sum_i x_ij = 1                 (grado de cliente)
       sum_j x_0j = sum_i x_i0 = K                      (rutas)
       t_j >= t_i + d_ij − M(1 − x_ij)   ∀ i, j∈clientes (MTZ; elimina subtours)
       t_j >= a_j                                       (espera a apertura)
       L_j >= t_j − b_j,  L_j >= 0                       (tardanza soft)
       sum_{i,j∈S} x_ij <= |S| − ⌈dem(S)/cap⌉   ∀S        (capacidad RCI)  [lazy]

La ruta a priori se evalúa con el mismo evaluador estocástico compartido
(``svrpx.stochastic``, con CRN), de modo que es directamente comparable con los
demás paradigmas.

Tamaño: la formulación dirigida tiene ~n² binarias; n<=~30 entra en la licencia
restringida de Gurobi, n=50 requiere licencia académica (se reporta el error con
claridad y el runner continúa).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from .._bootstrap import Instance, Solution, Solver, register_solver
from .. import stochastic
from .exact_bc import _components


@register_solver("exact-bc-tw")
class ExactBranchCutTW(Solver):
    """CVRPTW exacto por Branch & Cut dirigido (MTZ + ventanas soft) + evaluación estocástica."""

    def __init__(
        self,
        *,
        time_limit: float = 120.0,
        mip_gap: float = 0.0,
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        tw_penalty: float = 1.0,   # λ: peso de la tardanza nominal en el MIP
        threads: int = 0,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.tw_penalty = tw_penalty
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
        tw = np.asarray(instance.time_windows, dtype=float) if instance.time_windows is not None \
            else np.tile([0.0, stochastic._DAY], (n, 1))

        dmax = float(dist.max())
        T_ub = (n + 1) * (dmax + stochastic._DAY)  # cota superior de tiempo de llegada
        bigM = T_ub + dmax

        def k_of(S) -> int:
            dem = float(sum(demands[i] for i in S))
            return max(1, math.ceil(dem / cap)) if cap > 0 else 1

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
            gp.quicksum(float(dist[i, j]) * x[i, j] for (i, j) in x)
            + self.tw_penalty * gp.quicksum(L[h] for h in customers),
            GRB.MINIMIZE,
        )

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
                    m.addConstr(t[j] >= t[i] + float(dist[i, j]) - bigM * (1 - x[i, j]))

        cust_set = set(customers)
        conv_log: List[Tuple[float, float, float]] = []
        state = {"bst": None, "bnd": None}

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                xval = model.cbGetSolution(x)
                adj: Dict[int, List[int]] = {c: [] for c in customers}
                for (i, j), v in xval.items():
                    if v > 0.5 and i in cust_set and j in cust_set:
                        adj[i].append(j)
                        adj[j].append(i)
                for S in _components(adj, customers):
                    k = k_of(S)
                    cur = sum(xval[i, j] for i in S for j in S if i != j and (i, j) in xval)
                    if cur > len(S) - k + 0.5:
                        model.cbLazy(
                            gp.quicksum(x[i, j] for i in S for j in S if i != j) <= len(S) - k)
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
                f"Gurobi falló en n={n} (¿licencia/tamaño?): {e}. "
                "La formulación dirigida del CVRPTW (~n² binarias) requiere licencia "
                "académica para n>=~30."
            ) from e
        solve_time = time.time() - t0

        routes = self._extract_routes(x, m, depot, n, customers)
        mip_obj = float(m.ObjVal) if m.SolCount > 0 else float("nan")
        det_dist = float(sum(dist[i, j] * x[i, j].X for (i, j) in x)) if m.SolCount > 0 else float("nan")
        nominal_late = float(sum(L[h].X for h in customers)) if m.SolCount > 0 else float("nan")
        gap = float(m.MIPGap) if m.SolCount > 0 else float("nan")

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed,
            alpha=self.alpha, late_penalty=self.late_penalty, depot=depot,
        )

        extras = score.as_extras()
        extras.update({
            "det_cost": det_dist,
            "mip_obj": mip_obj,
            "nominal_tw_lateness": nominal_late,
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

    @staticmethod
    def _extract_routes(x, model, depot: int, n: int, customers) -> List[List[int]]:
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
