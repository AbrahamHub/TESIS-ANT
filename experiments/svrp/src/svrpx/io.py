"""Generación y carga de instancias TWCVRP de SVRPBench (depósito único).

Fidelidad al paquete oficial: se reutilizan directamente las primitivas oficiales
``city.City`` (muestreo de ubicaciones de clientes) y
``time_windows_generator.sample_time_window`` (ventanas residencial/comercial).
Se omite **sólo** el paso de ``city.Map`` que construye una rejilla de 1e6 puntos
y corre KMeans para ubicar el centro de ciudad: para todos nuestros tamaños
(10/20/50, ``num_cities = max(1, n//50) = 1``) ese centro es exactamente el
centroide del mapa, así que se calcula de forma cerrada (idéntico resultado,
~0.01 s en vez de ~23 s).

Normalización de capacidad: el código oficial fija ``capacity = sum(demandas)``
(1 vehículo cabe todo => capacidad no restrictiva). Para tener un CVRP genuino
—donde las desigualdades de capacidad redondeada del Branch & Cut sí se separan—
se fija una capacidad **restrictiva** según el paper (capacidad ≈ demanda_total /
num_vehículos), garantizando ``capacidad >= demanda_máxima`` para mantener
factibilidad. Queda documentado en ``instance.metadata``.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import _bootstrap  # asegura vrp_bench en el path
from ._bootstrap import Instance, SVRP_ROOT

# Primitivas oficiales (city sólo necesita numpy/sklearn/PIL, ya instalados).
from city import City, Location          # noqa: E402
from time_windows_generator import sample_time_window  # noqa: E402
from constants import MAP_SIZE, DEMAND_RANGE           # noqa: E402

DATA_DIR = SVRP_ROOT / "data" / "instances"


def _make_single_city(num_customers: int) -> City:
    """Réplica de ``city.Map`` para ``num_cities == 1``: una ciudad gaussiana
    centrada en el centroide del mapa, con el mismo rango de dispersión."""
    width, height = MAP_SIZE
    area = width * height
    spread = math.sqrt(area / (math.pi * 1))
    spread_range = (0.3 * spread, 0.4 * spread)
    cx = int(min(max(round((width - 1) / 2), 0), width - 1))
    cy = int(min(max(round((height - 1) / 2), 0), height - 1))
    city_spread = np.random.randint(int(spread_range[0]), int(spread_range[1]))
    return City((cx, cy), city_spread)


def _capacity(demands: np.ndarray, mode: str) -> Tuple[float, int, str]:
    """Capacidad y número de vehículos según ``mode``:

    * ``"binding"`` (default): ``cap = max(dem_máx, ⌈total/k_target⌉)`` con
      ~8 clientes por ruta y ``cap >= dem_máx`` (CVRP genuino → los cortes RCI del
      Branch & Cut se separan). **No** comparable 1:1 con SVRPBench oficial.
    * ``"official"``: ``cap = Σ demandas`` (no restrictiva), 1 vehículo — réplica
      exacta del generador oficial (`vehicle_capacity = sum(demands)`); el problema
      se reduce a mTSP con ventanas. Comparable con SVRPBench, pero la capacidad no
      restringe."""
    total = float(demands.sum())
    if mode == "official":
        return total, 1, "official cap = sum(demandas) (no restrictiva)"
    max_d = float(demands.max())
    n_customers = int((demands > 0).sum()) or len(demands)
    k_target = max(2, math.ceil(n_customers / 8))
    cap = max(max_d, math.ceil(total / k_target))
    num_vehicles = max(1, math.ceil(total / cap))
    return float(cap), int(num_vehicles), "binding cap = max(dem_máx, ceil(total/k_target))"


def generate_instance(num_customers: int, *, seed: int, capacity_mode: str = "binding") -> Instance:
    """Genera una instancia TWCVRP de depósito único, fiel a SVRPBench."""
    np.random.seed(seed)
    random.seed(seed)

    city = _make_single_city(num_customers)
    locs: List[Location] = city.batch_sample(MAP_SIZE, num_customers)
    customer_xy = np.array([(l.x, l.y) for l in locs], dtype=float)

    # Depósito único = centroide de los clientes (rama single-depot oficial).
    depot_xy = np.array([
        int(min(max(round(customer_xy[:, 0].mean()), 0), MAP_SIZE[0] - 1)),
        int(min(max(round(customer_xy[:, 1].mean()), 0), MAP_SIZE[1] - 1)),
    ], dtype=float)
    locations = np.vstack([depot_xy, customer_xy])  # depósito primero

    # Demandas U[0, 100]; depósito = 0 (DEMAND_RANGE oficial).
    demands = np.random.randint(DEMAND_RANGE[0], DEMAND_RANGE[1] + 1,
                                size=num_customers + 1).astype(float)
    demands[0] = 0.0

    # Ventanas de tiempo (residencial/comercial) con la primitiva oficial.
    appear_times = np.zeros(num_customers + 1, dtype=float)  # estático
    time_windows = np.empty((num_customers + 1, 2), dtype=float)
    time_windows[0] = (0.0, 1440.0)  # el depósito no tiene ventana
    for i in range(1, num_customers + 1):
        ctype = random.randint(0, 1)  # 0 residencial, 1 comercial
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
            "source": "svrpx.io.generate_instance",
            "variant": "twcvrp_single_depot",
            "num_customers": num_customers,
            "depot_index": 0,
            "seed": seed,
            "capacity_mode": capacity_mode,
            "capacity_note": cap_note,
        },
    )


def _npz_path(num_customers: int, n_instances: int, base_seed: int, capacity_mode: str) -> Path:
    return DATA_DIR / f"twcvrp_n{num_customers}_m{n_instances}_s{base_seed}_{capacity_mode}.npz"


def _save(insts: List[Instance], path: Path) -> None:
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


def _load(path: Path) -> List[Instance]:
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


def load_size(num_customers: int, n_instances: int = 3, *, base_seed: int = 12345,
              capacity_mode: str = "binding", cache: bool = True) -> List[Instance]:
    """Devuelve ``n_instances`` instancias de ``num_customers`` clientes,
    generándolas (y cacheándolas en .npz) si no existen."""
    path = _npz_path(num_customers, n_instances, base_seed, capacity_mode)
    if cache and path.exists():
        return _load(path)
    insts = [generate_instance(num_customers, seed=base_seed + 1000 * num_customers + k,
                               capacity_mode=capacity_mode)
             for k in range(n_instances)]
    if cache:
        _save(insts, path)
    return insts


def load_sizes(sizes: List[int], n_instances: int = 3, *, base_seed: int = 12345,
               capacity_mode: str = "binding", cache: bool = True) -> List[Tuple[int, Instance]]:
    """Lista de ``(size, Instance)`` para varios tamaños."""
    out: List[Tuple[int, Instance]] = []
    for s in sizes:
        for inst in load_size(s, n_instances, base_seed=base_seed,
                              capacity_mode=capacity_mode, cache=cache):
            out.append((s, inst))
    return out
