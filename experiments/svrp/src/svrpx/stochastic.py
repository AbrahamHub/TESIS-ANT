"""Evaluador estocástico compartido para los 5 paradigmas.

El modelo de tiempo de viaje estocástico (congestión por mezcla gaussiana,
retardo log-normal y accidentes de Poisson) y la simulación de ejecución de ruta
se **replican textualmente** de SVRPBench para producir números idénticos sin
arrastrar la cadena de imports pesada (``city`` -> ``scikit-learn`` + ``PIL``):

  * ``third_party/svrpbench/vrp_bench/travel_time_generator.py``
      -> normal_distribution, time_factor, random_factor, sample_accidents,
         calculate_delay, sample_travel_time
  * ``third_party/svrpbench/vrp_bench/vrp_base.py``
      -> _simulate_route_execution, _check_feasibility

Se añade, sobre esa base oficial:
  * recurso de 2ª etapa Q(ruta, ξ): penalización por minutos de retraso al violar
    una ventana de tiempo (formulación de programación estocástica en dos etapas);
  * CVaR_alpha del costo total c+Q (medida sensible al riesgo, anteproyecto §Metodología);
  * tasa de factibilidad y robustez agregadas sobre múltiples realizaciones,
    con semilla controlada para que todos los métodos vean los mismos escenarios.

Convención de rutas: ``routes`` es ``List[List[int]]`` con índices de cliente
(sin el depósito en los extremos), igual que ``vrp_bench.core.Solution``. El
depósito es el nodo 0 (instancias de depósito único generadas con depósito
primero); NO se infiere por ``demand == 0`` porque algunos clientes legítimos
pueden tener demanda 0 en SVRPBench.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Modelo de tiempo de viaje estocástico — copia textual de travel_time_generator.py
# (SVRPBench). Sólo depende de math/random/numpy.
# --------------------------------------------------------------------------- #


def normal_distribution(x: float, mean: float, std_dev: float) -> float:
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))


def time_factor(current_time: float) -> float:
    morning_peak = normal_distribution(current_time, 480, 90)
    evening_peak = normal_distribution(current_time, 1020, 90)
    return 0.5 + 2 * (morning_peak + evening_peak)


def random_factor(current_time: float) -> float:
    rush_hour_effect = normal_distribution(current_time, 480, 90) + normal_distribution(current_time, 1020, 90)
    mu = 0 + 0.1 * rush_hour_effect
    sigma = 0.3 + 0.2 * rush_hour_effect
    return random.lognormvariate(mu, sigma)


def sample_accidents(current_time: float) -> int:
    accident_rate = 0.05 * normal_distribution(current_time, 1260, 120)  # pico a las 21:00
    if accident_rate < 0:
        accident_rate = 0
    return np.random.poisson(lam=accident_rate)


def calculate_delay(distance: float, current_time: float) -> float:
    time_fac = time_factor(current_time)
    distance_factor = 1 - math.exp(-distance / 50)
    base_delay = 0.25 * time_fac * distance_factor
    rand_factor = random_factor(current_time)
    delay = base_delay * rand_factor

    num_accidents = sample_accidents(current_time)
    accident_delay = 0
    if num_accidents > 0:
        durations = np.random.uniform(30, 120, size=num_accidents)
        accident_delay = np.sum(durations)
    delay += accident_delay
    return delay


def sample_travel_time(a: int, b: int, distances: Dict[Tuple[int, int], float],
                       current_time: float, velocity: float = 1) -> float:
    if a == b:
        return 0.0
    distance = distances[(a, b)]
    delay = calculate_delay(distance, current_time)
    return distance / velocity + delay


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #


def euclidean_int_matrix(locations: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclidianas **enteras** (como ``city.Location.distance``,
    que hace ``np.linalg.norm(...).astype(int)``). Es la métrica determinista que
    usan tanto el MIP exacto como la simulación estocástica."""
    locs = np.asarray(locations, dtype=np.float64)
    diff = locs[:, None, :] - locs[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return d.astype(int).astype(np.float64)


def distances_dict(dist_matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    n = dist_matrix.shape[0]
    return {(i, j): float(dist_matrix[i, j]) for i in range(n) for j in range(n)}


# --------------------------------------------------------------------------- #
# Simulación de ejecución de ruta (replica vrp_base._simulate_route_execution /
# _check_feasibility, extendida con recurso de 2ª etapa y conteo de retrasos).
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioResult:
    travel_cost: float          # tiempo total de viaje (costo c, comparable con SVRPBench)
    waiting: float              # tiempo de espera por ventanas/aparición
    recourse: float             # Q(ruta, ξ): penalización por retrasos (2ª etapa)
    tw_violations: int          # número de ventanas de tiempo violadas
    feasible: bool              # 1 si sin violaciones (capacidad+TW+duplicados+no servidos)
    violations: int             # total de violaciones (para CVR)


def _simulate_one_scenario(
    routes: Sequence[Sequence[int]],
    depot: int,
    distances: Dict[Tuple[int, int], float],
    demands: np.ndarray,
    capacities: Sequence[float],
    time_windows: Dict[int, Tuple[float, float]],
    appear_times: Dict[int, float],
    customers: set,
    late_penalty: float,
) -> ScenarioResult:
    """Una realización estocástica: recorre cada ruta con tiempos de viaje
    muestreados y acumula costo, espera, recurso y violaciones."""
    total_cost = 0.0
    total_wait = 0.0
    total_recourse = 0.0
    tw_violations = 0
    capacity_violations = 0
    appear_violations = 0
    visit_count: Dict[int, int] = {}
    served = 0

    for r_idx, raw in enumerate(routes):
        # Acotar la ruta con el depósito en ambos extremos.
        route = [depot] + [int(c) for c in raw] + [depot]
        if len(route) <= 2:
            continue
        cap = capacities[r_idx] if r_idx < len(capacities) else capacities[0]
        route_demand = 0.0
        current_time = 0.0

        for i in range(len(route) - 1):
            cur, nxt = route[i], route[i + 1]
            if nxt in customers:
                served += 1
                visit_count[nxt] = visit_count.get(nxt, 0) + 1
                if nxt < len(demands):
                    route_demand += float(demands[nxt])

            travel = sample_travel_time(cur, nxt, distances, current_time)
            current_time += travel
            total_cost += travel

            # Aparición dinámica del cliente (espera si llega antes).
            if nxt in appear_times and current_time < appear_times[nxt]:
                total_wait += appear_times[nxt] - current_time
                current_time = appear_times[nxt]

            # Ventana de tiempo (recurso de 2ª etapa al violar el cierre).
            if nxt in time_windows and nxt != depot:
                start, end = time_windows[nxt]
                t_norm = current_time % 1440  # semántica oficial de SVRPBench
                if t_norm > end:
                    tw_violations += 1
                    total_recourse += late_penalty * (t_norm - end)
                elif t_norm < start:
                    total_wait += start - t_norm
                    current_time += start - t_norm

        if route_demand > cap * 1.001:
            capacity_violations += 1

    violations = capacity_violations + tw_violations + appear_violations
    for _c, cnt in visit_count.items():
        if cnt > 1:
            violations += cnt - 1
    violations += max(0, len(customers) - served)

    return ScenarioResult(
        travel_cost=total_cost,
        waiting=total_wait,
        recourse=total_recourse,
        tw_violations=tw_violations,
        feasible=(violations == 0),
        violations=violations,
    )


# --------------------------------------------------------------------------- #
# CVaR y agregación
# --------------------------------------------------------------------------- #


def cvar(samples: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR_alpha (lado de pérdidas): media del peor ``(1-alpha)`` de los costos."""
    s = np.sort(np.asarray(samples, dtype=np.float64))
    if s.size == 0:
        return 0.0
    k = max(1, int(math.ceil((1.0 - alpha) * s.size)))
    return float(s[-k:].mean())


@dataclass
class StochasticScore:
    expected_cost: float          # E[c] tiempo de viaje (comparable SVRPBench)
    expected_total: float         # E[c + Q] costo con recurso
    cvar: float                   # CVaR_alpha(c + Q)
    feasibility: float            # tasa de factibilidad en [0, 1]
    cvr: float                    # tasa de violación de restricciones (%)
    waiting_time: float
    robustness: float             # desviación estándar de c entre realizaciones
    tw_violations: float          # promedio de violaciones de ventana
    alpha: float
    cost_samples: np.ndarray = field(repr=False, default=None)
    total_samples: np.ndarray = field(repr=False, default=None)

    def as_extras(self) -> dict:
        return {
            "expected_cost": self.expected_cost,
            "expected_total": self.expected_total,
            "cvar": self.cvar,
            "tw_violations": self.tw_violations,
            "alpha": self.alpha,
        }


def score_routes(
    instance,
    routes: List[List[int]],
    *,
    num_realizations: int = 200,
    seed: int = 0,
    alpha: float = 0.95,
    late_penalty: float = 1.0,
    depot: int = 0,
) -> StochasticScore:
    """Puntúa rutas (índices de cliente, sin depósito) sobre ``num_realizations``
    escenarios estocásticos. Semilla fija por instancia => todos los métodos ven
    los mismos escenarios (comparabilidad)."""
    locations = np.asarray(instance.locations, dtype=np.float64)
    demands = np.asarray(instance.demands, dtype=np.float64)
    n = locations.shape[0]
    dist = euclidean_int_matrix(locations)
    dd = distances_dict(dist)

    caps = list(np.asarray(instance.vehicle_capacities, dtype=np.float64).ravel())
    if not caps:
        caps = [float(demands.sum())]

    tw: Dict[int, Tuple[float, float]] = {}
    if getattr(instance, "time_windows", None) is not None:
        twa = np.asarray(instance.time_windows, dtype=np.float64)
        for i in range(min(n, twa.shape[0])):
            tw[i] = (float(twa[i, 0]), float(twa[i, 1]))

    appear: Dict[int, float] = {}
    if getattr(instance, "appear_times", None) is not None:
        ap = np.asarray(instance.appear_times, dtype=np.float64).ravel()
        for i in range(min(n, ap.shape[0])):
            if ap[i] > 0:
                appear[i] = float(ap[i])

    customers = set(range(n)) - {depot}

    costs = np.empty(num_realizations)
    totals = np.empty(num_realizations)
    waits = np.empty(num_realizations)
    feas = np.empty(num_realizations)
    cvrs = np.empty(num_realizations)
    twv = np.empty(num_realizations)

    n_customers = max(1, len(customers))
    for r in range(num_realizations):
        s = (seed * 100003 + r) & 0x7FFFFFFF
        random.seed(s)
        np.random.seed(s)
        res = _simulate_one_scenario(
            routes, depot, dd, demands, caps, tw, appear, customers, late_penalty
        )
        costs[r] = res.travel_cost
        totals[r] = res.travel_cost + res.recourse
        waits[r] = res.waiting
        feas[r] = 1.0 if res.feasible else 0.0
        cvrs[r] = (res.violations / n_customers) * 100.0
        twv[r] = res.tw_violations

    return StochasticScore(
        expected_cost=float(costs.mean()),
        expected_total=float(totals.mean()),
        cvar=cvar(totals, alpha),
        feasibility=float(feas.mean()),
        cvr=float(cvrs.mean()),
        waiting_time=float(waits.mean()),
        robustness=float(costs.std()),
        tw_violations=float(twv.mean()),
        alpha=alpha,
        cost_samples=costs,
        total_samples=totals,
    )
