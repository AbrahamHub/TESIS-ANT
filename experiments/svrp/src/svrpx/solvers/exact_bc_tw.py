"""Baseline exacto CVRPTW (Branch & Cut dirigido + MTZ con ventanas de tiempo).

Complementa a ``exact-bc`` (que ignora las ventanas) para **aislar el efecto de la
estocasticidad**: este solver SÍ intenta respetar las ventanas (deterministas) en el
escenario nominal. Para que ese horario nominal sea realista (no optimista), tanto el
objetivo como la propagación MTZ usan el **tiempo de viaje nominal**
``τ_ij = d_ij + retraso_de_congestión_determinista(t*)`` —no solo la distancia—, de
modo que la congestión conocida se incorpora al plan y la infactibilidad residual bajo
ξ es atribuible a la incertidumbre, no a un horario subestimado.

  min  sum_{i!=j} τ_ij x_ij  +  λ · sum_i L_i
  s.a. sum_j x_ij = 1,  sum_i x_ij = 1,  sum_j x_0j = sum_i x_i0 = K
       t_j >= t_i + τ_ij − M(1 − x_ij),  t_j >= a_j,  L_j >= t_j − b_j     (MTZ + ventanas soft)
       sum_{i,j∈S} x_ij <= |S| − ⌈dem(S)/cap⌉   ∀S                          (RCI; lazy + user cuts)

Tamaño: ~n² binarias; n<=~30 entra en la licencia restringida de Gurobi, n=50 requiere
licencia académica (el runner lo salta con un aviso claro).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from .._bootstrap import Instance, Solution, Solver, register_solver
from .. import stochastic
from .exact_bc import violated_rci


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
        tw_penalty: float = 1.0,
        accident_scale: float = 1.0,
        frac_sep_until_node: int = 200,
        threads: int = 0,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.tw_penalty = tw_penalty
        self.accident_scale = accident_scale
        self.frac_sep_until_node = frac_sep_until_node
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
        tau = stochastic.nominal_time_matrix(dist, t_star)  # tiempo nominal (objetivo + MTZ)
        tw = np.asarray(instance.time_windows, dtype=float) if instance.time_windows is not None \
            else np.tile([0.0, stochastic._DAY], (n, 1))

        T_ub = (n + 1) * (float(tau.max()) + stochastic._DAY)
        bigM = T_ub + float(tau.max())

        m = gp.Model("exact_bc_cvrptw")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.Threads = self.threads
        m.Params.LazyConstraints = 1
        m.Params.PreCrush = 1

        x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
             for i in range(n) for j in range(n) if i != j}
        t = {i: m.addVar(lb=0.0, ub=T_ub, name=f"t_{i}") for i in range(n)}
        L = {h: m.addVar(lb=0.0, name=f"L_{h}") for h in customers}

        m.setObjective(
            gp.quicksum(float(tau[i, j]) * x[i, j] for (i, j) in x)
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
                    m.addConstr(t[j] >= t[i] + float(tau[i, j]) - bigM * (1 - x[i, j]))

        conv_log: List[Tuple[float, float, float]] = []
        state = {"bst": None, "bnd": None}

        def add_cut(model, S, k, lazy):
            expr = gp.quicksum(x[i, j] for i in S for j in S if i != j)
            (model.cbLazy if lazy else model.cbCut)(expr <= len(S) - k)

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                xval = model.cbGetSolution(x)
                for S, k in violated_rci(customers, demands, cap,
                                         lambda i, j: xval[i, j] + xval[j, i], eps=0.5):
                    add_cut(model, S, k, lazy=True)
            elif where == GRB.Callback.MIPNODE:
                if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if model.cbGet(GRB.Callback.MIPNODE_NODCNT) > self.frac_sep_until_node:
                    return
                xrel = model.cbGetNodeRel(x)
                for S, k in violated_rci(customers, demands, cap,
                                         lambda i, j: xrel[i, j] + xrel[j, i], thr=1e-4, eps=1e-3):
                    add_cut(model, S, k, lazy=False)
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

        routes = self._extract_routes(x, m, depot, n)
        mip_obj = float(m.ObjVal) if m.SolCount > 0 else float("nan")
        det_time = float(sum(tau[i, j] * x[i, j].X for (i, j) in x)) if m.SolCount > 0 else float("nan")
        nominal_late = float(sum(L[h].X for h in customers)) if m.SolCount > 0 else float("nan")
        gap = float(m.MIPGap) if m.SolCount > 0 else float("nan")

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot,
        )

        extras = score.as_extras()
        extras.update({
            "det_cost": det_time,
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
