"""Evaluador estocástico compartido para los 5 paradigmas.

Fidelidad a SVRPBench (revisión v3):

  * **Primitivas idénticas**: el modelo de tiempo de viaje (congestión por mezcla
    gaussiana con picos 8:00/17:00, factor log-normal y accidentes de Poisson con
    pico a las 21:00) reproduce exactamente las fórmulas de
    ``travel_time_generator.py``. El factor aleatorio log-normal se expresa como
    ``exp(mu(t) + sigma(t)·Z)`` con ``Z ~ N(0,1)``, equivalente en distribución a
    ``random.lognormvariate(mu, sigma)``. Los accidentes suman **exactamente**
    ``cnt`` duraciones Uniforme(30,120) i.i.d. (no una aproximación).

  * **Semántica de costo idéntica** a ``vrp_base._simulate_route_execution``:
    ``current_time`` crudo (sin módulo) para muestrear el siguiente tramo; en
    llegada temprana se espera (``current_time = inicio_ventana``); el costo
    acumula **solo tiempo de viaje**. Violaciones de ventana en hora-del-día
    (``% 1440``) y de aparición (``current_time < appear``), como en
    ``vrp_base._check_feasibility``.

  * **Common Random Numbers (CRN) reales**: cada realización ``r`` pre-muestrea un
    **escenario** ``ξ`` —ruido log-normal y demora total por accidentes por
    (arco, bucket horario)— *independiente de la ruta* y determinista en
    ``(seed, r)``. Dos métodos cualesquiera sobre la misma instancia ven el mismo ξ
    en la realización ``r`` (CRN; varianza reducida; pruebas estadísticas válidas).

  * **Extensión sobre la base oficial**: recurso de 2ª etapa ``Q(ruta, ξ)``
    (penalización ``late_penalty`` por minuto de retraso al violar una ventana) y
    ``CVaR_alpha`` del costo total ``c + Q``.

  * **Control de cola** (``accident_scale``): multiplica la tasa de accidentes de
    Poisson. Con el valor oficial (1.0) los accidentes son rarísimos (≈1e-4) y el
    CVaR ≈ media; subirlo (p. ej. 20-50) hace que ξ produzca colas pesadas reales y
    el CVaR/robustez discriminen. Default 1.0 (fiel a SVRPBench).

Diferencia consciente con el evaluador oficial: SVRPBench muestrea costo y
factibilidad con realizaciones independientes; aquí ambos usan el **mismo** ξ (más
sólido y necesario para el CRN). ``expected_cost`` comparte la *semántica* del costo
oficial pero no es bit-a-bit idéntico.

Convención de rutas: ``routes`` es ``List[List[int]]`` con índices de cliente (sin el
depósito en los extremos), igual que ``vrp_bench.core.Solution``. El depósito es el
nodo 0; NO se infiere por ``demand == 0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

_DAY = 1440.0

# --------------------------------------------------------------------------- #
# Partes deterministas del modelo (copia de travel_time_generator.py).
# --------------------------------------------------------------------------- #


def normal_distribution(x: float, mean: float, std_dev: float) -> float:
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))


def time_factor(current_time: float) -> float:
    morning_peak = normal_distribution(current_time, 480, 90)
    evening_peak = normal_distribution(current_time, 1020, 90)
    return 0.5 + 2 * (morning_peak + evening_peak)


def _rush(current_time: float) -> float:
    return normal_distribution(current_time, 480, 90) + normal_distribution(current_time, 1020, 90)


# --------------------------------------------------------------------------- #
# Geometría y tiempo nominal (determinista) — usado por los MIP exactos.
# --------------------------------------------------------------------------- #


def euclidean_int_matrix(locations: np.ndarray) -> np.ndarray:
    """Distancias euclidianas **enteras** (como ``city.Location.distance``)."""
    locs = np.asarray(locations, dtype=np.float64)
    diff = locs[:, None, :] - locs[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(int).astype(np.float64)


def nominal_time_matrix(dist: np.ndarray, t_star: float) -> np.ndarray:
    """Tiempo de viaje **nominal determinista** a una hora representativa ``t_star``:
    ``τ_ij = d_ij + 0.25·time_factor(t_star)·(1 − e^{−d_ij/50})`` (velocidad=1,
    factor log-normal en su mediana ≈ 1, sin accidentes). Es la parte conocida del
    tiempo de viaje; los MIP exactos optimizan sobre ``τ`` (objetivo de tiempo de
    viaje de SVRPBench) y propagan el horario MTZ con ``τ``, evitando un plan
    optimista que ignore la congestión determinista."""
    d = np.asarray(dist, dtype=np.float64)
    base = 0.25 * time_factor(t_star) * (1.0 - np.exp(-d / 50.0))
    tau = d + base
    np.fill_diagonal(tau, 0.0)
    return tau


def representative_time(instance, depot: int = 0) -> float:
    """Hora representativa de servicio = media de los centros de ventana de los
    clientes (fallback: mediodía)."""
    tw = getattr(instance, "time_windows", None)
    if tw is None:
        return 720.0
    tw = np.asarray(tw, dtype=np.float64)
    centers = [(tw[i, 0] + tw[i, 1]) / 2.0 for i in range(tw.shape[0]) if i != depot]
    return float(np.mean(centers)) if centers else 720.0


# --------------------------------------------------------------------------- #
# Escenario pre-muestreado ξ (CRN). Ruido independiente de la ruta.
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    z: np.ndarray          # (n, n, B) ~ N(0,1): ruido log-normal por (arco, bucket)
    acc_delay: np.ndarray  # (n, n, B): demora TOTAL por accidentes (suma exacta de cnt Uniformes)
    n_buckets: int


def sample_scenario(n: int, base_seed: int, r: int, n_buckets: int = 24,
                    accident_scale: float = 1.0) -> Scenario:
    """Pre-muestrea ξ para la realización ``r``, determinista en ``(seed, r)``."""
    rng = np.random.default_rng((base_seed * 1_000_003 + r) & 0x7FFFFFFFFFFFFFFF)
    B = n_buckets
    z = rng.standard_normal((n, n, B))

    mids = (np.arange(B) + 0.5) * (_DAY / B)
    rate = 0.05 * accident_scale * np.array([normal_distribution(m, 1260, 120) for m in mids])
    rate = np.clip(rate, 0.0, None)
    cnt = rng.poisson(lam=np.broadcast_to(rate, (n, n, B)))

    acc_delay = np.zeros((n, n, B), dtype=np.float64)
    mx = int(cnt.max())
    if mx > 0:  # suma exacta de cnt duraciones i.i.d. Uniforme(30,120)
        dur = rng.uniform(30.0, 120.0, size=(n, n, B, mx))
        mask = np.arange(mx)[None, None, None, :] < cnt[..., None]
        acc_delay = (dur * mask).sum(-1)
    return Scenario(z=z, acc_delay=acc_delay, n_buckets=B)


def _bucket(t: float, B: int) -> int:
    return int((t % _DAY) // (_DAY / B))


def scenario_travel_time(i: int, j: int, t: float, dist: np.ndarray, scen: Scenario) -> float:
    """Tiempo de viaje muestreado bajo el escenario fijo ξ. Determinista en (i, j, t, ξ)."""
    if i == j:
        return 0.0
    d = float(dist[i, j])
    b = _bucket(t, scen.n_buckets)
    base_delay = 0.25 * time_factor(t) * (1.0 - math.exp(-d / 50.0))
    rush = _rush(t)
    mu = 0.0 + 0.1 * rush
    sigma = 0.3 + 0.2 * rush
    delay = base_delay * math.exp(mu + sigma * float(scen.z[i, j, b]))
    delay += float(scen.acc_delay[i, j, b])
    return d / 1.0 + delay


# --------------------------------------------------------------------------- #
# Simulación de una realización (semántica oficial de costo + recurso de 2ª etapa).
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioResult:
    travel_cost: float
    waiting: float
    recourse: float
    tw_violations: int
    feasible: bool
    violations: int


def _simulate(
    routes: Sequence[Sequence[int]],
    depot: int,
    dist: np.ndarray,
    demands: np.ndarray,
    capacities: Sequence[float],
    time_windows: Dict[int, Tuple[float, float]],
    appear_times: Dict[int, float],
    customers: set,
    late_penalty: float,
    scen: Scenario,
) -> ScenarioResult:
    total_cost = total_wait = total_recourse = 0.0
    tw_violations = capacity_violations = appear_violations = 0
    visit: Dict[int, int] = {}
    served = 0

    for r_idx, raw in enumerate(routes):
        route = [depot] + [int(c) for c in raw] + [depot]
        if len(route) <= 2:
            continue
        cap = capacities[r_idx] if r_idx < len(capacities) else capacities[0]
        route_demand = 0.0
        ct = 0.0  # current_time crudo (semántica oficial de costo)

        for k in range(len(route) - 1):
            cur, nxt = route[k], route[k + 1]
            if nxt in customers:
                served += 1
                visit[nxt] = visit.get(nxt, 0) + 1
                if nxt < len(demands):
                    route_demand += float(demands[nxt])

            travel = scenario_travel_time(cur, nxt, ct, dist, scen)
            ct += travel
            total_cost += travel

            if nxt in appear_times and ct < appear_times[nxt]:
                appear_violations += 1  # llegar antes de que el cliente aparezca
                total_wait += appear_times[nxt] - ct
                ct = appear_times[nxt]

            if nxt in time_windows and nxt != depot:
                start, end = time_windows[nxt]
                if ct < start:  # espera por llegada temprana (semántica oficial, crudo)
                    total_wait += start - ct
                    ct = start
                t_norm = ct % _DAY  # violación + recurso (hora-del-día, semántica oficial)
                if t_norm > end:
                    tw_violations += 1
                    total_recourse += late_penalty * (t_norm - end)

        if route_demand > cap * 1.001:
            capacity_violations += 1

    violations = capacity_violations + tw_violations + appear_violations
    for _c, cnt in visit.items():
        if cnt > 1:
            violations += cnt - 1
    violations += max(0, len(customers) - served)

    return ScenarioResult(total_cost, total_wait, total_recourse, tw_violations,
                          violations == 0, violations)


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
    expected_cost: float          # E[c]  tiempo de viaje (semántica SVRPBench)
    expected_total: float         # E[c + Q]  costo con recurso de 2ª etapa
    cvar: float                   # CVaR_alpha(c + Q)
    feasibility: float            # tasa de factibilidad en [0, 1]
    cvr: float                    # tasa de violación de restricciones (%)
    waiting_time: float
    robustness: float             # desviación estándar de c entre realizaciones
    tw_violations: float
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
    accident_scale: float = 1.0,
    vehicle_fixed_cost: float = 0.0,
    depot: int = 0,
    n_buckets: int = 24,
) -> StochasticScore:
    """Puntúa rutas sobre ``num_realizations`` escenarios pre-muestreados (CRN).
    Para una misma instancia (mismo ``seed``) todos los métodos ven escenarios
    idénticos, independientemente de sus rutas.

    ``vehicle_fixed_cost`` (homologación de flota): costo fijo por vehículo/ruta
    usado, sumado de forma **uniforme** a todos los métodos. Internaliza el tradeoff
    "factibilidad a cambio de más vehículos" en el costo comparable (un método que usa
    12 rutas paga 12·c_fijo; uno que usa 4 paga 4·c_fijo) → igualdad de condiciones."""
    locations = np.asarray(instance.locations, dtype=np.float64)
    demands = np.asarray(instance.demands, dtype=np.float64)
    n = locations.shape[0]
    dist = euclidean_int_matrix(locations)

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
    n_customers = max(1, len(customers))

    costs = np.empty(num_realizations)
    totals = np.empty(num_realizations)
    waits = np.empty(num_realizations)
    feas = np.empty(num_realizations)
    cvrs = np.empty(num_realizations)
    twv = np.empty(num_realizations)

    for r in range(num_realizations):
        scen = sample_scenario(n, seed, r, n_buckets=n_buckets, accident_scale=accident_scale)
        res = _simulate(routes, depot, dist, demands, caps, tw, appear, customers,
                        late_penalty, scen)
        costs[r] = res.travel_cost
        totals[r] = res.travel_cost + res.recourse
        waits[r] = res.waiting
        feas[r] = 1.0 if res.feasible else 0.0
        cvrs[r] = (res.violations / n_customers) * 100.0
        twv[r] = res.tw_violations

    # Homologación de flota: costo fijo por ruta usada (determinista, uniforme).
    if vehicle_fixed_cost:
        fleet_cost = sum(1 for rt in routes if len(rt) > 0) * float(vehicle_fixed_cost)
        costs += fleet_cost
        totals += fleet_cost

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
