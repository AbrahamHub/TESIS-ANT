"""Generación y carga del **banco canónico** de instancias SVRP.

Punto crítico para el "piso parejo": los cinco paradigmas resuelven exactamente las
mismas instancias. El banco se construye **una vez** (notebook 00) con semillas fijas,
se cachea en ``.npz`` y los cinco notebooks lo cargan con ``load_bank(...)``.

Fidelidad a SVRPBench: se reutilizan **las primitivas oficiales** del repo clonado
—``city.City.batch_sample`` (ubicaciones gaussianas alrededor de un centro de ciudad)
y ``time_windows_generator.sample_time_window`` (ventanas residencial/comercial)—,
exactamente como el generador del benchmark. Se omite **solo** el paso de ``city.Map``
que construye una rejilla de 10^6 puntos y corre KMeans para situar el centro de ciudad:
para depósito único (``num_cities = max(1, n//50)``; aquí trabajamos con 1 ciudad por
instancia para escalas ≤ 300) ese centro es el centroide del mapa, así que se calcula de
forma cerrada (resultado idéntico, ~0.01 s en vez de ~23 s).

Capacidad: el generador oficial fija ``capacity = sum(demandas)`` (1 vehículo cabe todo
⇒ capacidad no restrictiva). Para un CVRP genuino —donde los cortes de capacidad del
Branch & Cut sí se separan y el problema discrimina entre métodos— se ofrece
``capacity_mode="binding"`` (``cap = max(dem_máx, ⌈total/k⌉)``). ``"official"`` replica
la regla oficial. La elección queda en ``instance.metadata``.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# La API oficial se importa de forma diferida (requiere bootstrap.init() antes).


def _official():
    """Importa las primitivas oficiales (city, ventanas, constantes, Instance)."""
    from vrp_bench.core.instance import Instance  # noqa
    from city import City, Location               # noqa
    from time_windows_generator import sample_time_window  # noqa
    try:
        from constants import MAP_SIZE, DEMAND_RANGE  # noqa
    except Exception:  # algunos clones no exponen constants en el path plano
        MAP_SIZE, DEMAND_RANGE = (1000, 1000), (0, 100)
    return Instance, City, Location, sample_time_window, MAP_SIZE, DEMAND_RANGE


def _make_single_city(City, MAP_SIZE):
    """Réplica de ``city.Map`` para ``num_cities == 1``: una ciudad gaussiana centrada
    en el centroide del mapa, con el mismo rango de dispersión que el oficial."""
    width, height = MAP_SIZE
    area = width * height
    spread = math.sqrt(area / (math.pi * 1))
    spread_range = (0.3 * spread, 0.4 * spread)
    cx = int(min(max(round((width - 1) / 2), 0), width - 1))
    cy = int(min(max(round((height - 1) / 2), 0), height - 1))
    city_spread = np.random.randint(int(spread_range[0]), int(spread_range[1]))
    return City((cx, cy), city_spread)


def _capacity(demands: np.ndarray, mode: str) -> Tuple[float, int, str]:
    total = float(demands.sum())
    if mode == "official":
        return total, 1, "official cap = sum(demandas) (no restrictiva)"
    max_d = float(demands.max())
    n_customers = int((demands > 0).sum()) or len(demands)
    k_target = max(2, math.ceil(n_customers / 8))   # ~8 clientes por ruta
    cap = max(max_d, math.ceil(total / k_target))
    num_vehicles = max(1, math.ceil(total / cap))
    return float(cap), int(num_vehicles), "binding cap = max(dem_máx, ceil(total/k_target))"


def generate_instance(num_customers: int, *, seed: int, capacity_mode: str = "binding"):
    """Genera una instancia TWCVRP de depósito único, fiel a SVRPBench."""
    Instance, City, Location, sample_time_window, MAP_SIZE, DEMAND_RANGE = _official()
    np.random.seed(seed)
    random.seed(seed)

    city = _make_single_city(City, MAP_SIZE)
    locs = city.batch_sample(MAP_SIZE, num_customers)
    customer_xy = np.array([(l.x, l.y) for l in locs], dtype=float)

    depot_xy = np.array([
        int(min(max(round(customer_xy[:, 0].mean()), 0), MAP_SIZE[0] - 1)),
        int(min(max(round(customer_xy[:, 1].mean()), 0), MAP_SIZE[1] - 1)),
    ], dtype=float)
    locations = np.vstack([depot_xy, customer_xy])   # depósito primero

    # Demandas U[1, 100]; depósito = 0. Cota inferior 1 (no la 0 oficial) para que el
    # ÚNICO nodo con demanda 0 sea el depósito (los solvers heredados lo detectan así).
    low = max(1, DEMAND_RANGE[0])
    demands = np.random.randint(low, DEMAND_RANGE[1] + 1, size=num_customers + 1).astype(float)
    demands[0] = 0.0

    appear_times = np.zeros(num_customers + 1, dtype=float)   # estático
    time_windows = np.empty((num_customers + 1, 2), dtype=float)
    time_windows[0] = (0.0, 1440.0)                            # depósito sin ventana
    for i in range(1, num_customers + 1):
        ctype = random.randint(0, 1)   # 0 residencial, 1 comercial
        time_windows[i] = sample_time_window(ctype, appear_times[i])

    cap, num_vehicles, cap_note = _capacity(demands, capacity_mode)

    return Instance(
        locations=locations,
        demands=demands,
        vehicle_capacities=np.full(num_vehicles, cap, dtype=float),
        num_vehicles=num_vehicles,
        time_windows=time_windows,
        appear_times=appear_times,
        metadata={
            "source": "svrplab.data.generate_instance",
            "variant": "twcvrp_single_depot",
            "num_customers": num_customers,
            "depot_index": 0,
            "seed": seed,
            "capacity_mode": capacity_mode,
            "capacity_note": cap_note,
        },
    )


# --------------------------------------------------------------------------- #
# Banco canónico: persistencia .npz
# --------------------------------------------------------------------------- #


def _instance_seed(base_seed: int, num_customers: int, k: int) -> int:
    """Semilla determinista por (banco, tamaño, índice). Estable entre notebooks."""
    return base_seed + 1000 * num_customers + k


def _bank_path(data_dir: Path, num_customers: int, n_instances: int,
               base_seed: int, capacity_mode: str) -> Path:
    return data_dir / f"bank_n{num_customers}_m{n_instances}_s{base_seed}_{capacity_mode}.npz"


def _save_bank(insts: List, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        locations=np.array([i.locations for i in insts], dtype=object),
        demands=np.array([i.demands for i in insts], dtype=object),
        vehicle_capacities=np.array([i.vehicle_capacities for i in insts], dtype=object),
        num_vehicles=np.array([i.num_vehicles for i in insts]),
        time_windows=np.array([i.time_windows for i in insts], dtype=object),
        appear_times=np.array([i.appear_times for i in insts], dtype=object),
        metadata=np.array([i.metadata for i in insts], dtype=object),
    )


def _load_bank(path: Path) -> List:
    Instance = _official()[0]
    raw = np.load(path, allow_pickle=True)
    out = []
    for k in range(len(raw["locations"])):
        out.append(Instance(
            locations=np.asarray(raw["locations"][k], dtype=float),
            demands=np.asarray(raw["demands"][k], dtype=float),
            vehicle_capacities=np.asarray(raw["vehicle_capacities"][k], dtype=float),
            num_vehicles=int(raw["num_vehicles"][k]),
            time_windows=np.asarray(raw["time_windows"][k], dtype=float),
            appear_times=np.asarray(raw["appear_times"][k], dtype=float),
            metadata=dict(raw["metadata"][k]),
        ))
    return out


def build_bank(data_dir, sizes: List[int], n_instances: int, *,
               base_seed: int = 12345, capacity_mode: str = "binding",
               verbose: bool = True) -> dict:
    """Construye (y cachea) el banco canónico para varios tamaños. Idempotente:
    si el ``.npz`` existe, no regenera. Devuelve ``{size: [Instance, ...]}``."""
    data_dir = Path(data_dir)
    bank = {}
    for s in sizes:
        path = _bank_path(data_dir, s, n_instances, base_seed, capacity_mode)
        if path.exists():
            insts = _load_bank(path)
            if verbose:
                print(f"[data] n={s}: {len(insts)} instancias (cache) -> {path.name}")
        else:
            insts = [generate_instance(s, seed=_instance_seed(base_seed, s, k),
                                       capacity_mode=capacity_mode)
                     for k in range(n_instances)]
            _save_bank(insts, path)
            if verbose:
                print(f"[data] n={s}: {len(insts)} instancias generadas -> {path.name}")
        bank[s] = insts
    return bank


def load_bank(data_dir, sizes: List[int], n_instances: int, *,
              base_seed: int = 12345, capacity_mode: str = "binding",
              verbose: bool = False) -> dict:
    """Carga el banco canónico (lo construye si falta). Misma firma que ``build_bank``."""
    return build_bank(data_dir, sizes, n_instances, base_seed=base_seed,
                      capacity_mode=capacity_mode, verbose=verbose)


def bank_as_pairs(bank: dict) -> List[Tuple[int, object]]:
    """Aplana ``{size: [inst,...]}`` a ``[(size, inst), ...]`` (orden por tamaño)."""
    out = []
    for s in sorted(bank):
        for inst in bank[s]:
            out.append((s, inst))
    return out
