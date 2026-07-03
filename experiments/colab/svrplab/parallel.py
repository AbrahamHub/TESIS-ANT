"""Paralelismo multi-proceso (fork) para las fases CPU-bound del pipeline.

En Colab la GPU convive con pocas vCPU (T4: 2, L4: 8, A100: 12). Los cuellos de
botella CPU del pipeline —Gurobi, ACO/Tabu heredados y el evaluador CRN (bucle
Python sobre tramos de ruta)— NO escalan con hilos por el GIL; escalan con
**procesos**. Se usa el contexto ``fork`` (Linux/Colab): los workers heredan
``sys.path`` y los módulos ya importados (incluido el repo oficial de SVRPBench)
por copy-on-write, sin re-import ni serialización del entorno.

Regla de seguridad: los workers ejecutan SOLO numpy/Gurobi/código Python puro.
Nunca tocan CUDA (forkear un proceso con contexto CUDA es seguro mientras el
hijo no llame a la GPU); por eso ``runner.run_solver`` mantiene secuencial la
fase de *solve* de los solvers GPU y paraleliza únicamente la re-puntuación.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, List, Sequence


def effective_jobs(n_jobs: int | None) -> int:
    """Normaliza ``n_jobs``: None/0/1 → 1 (secuencial); -1 → nº de CPUs."""
    if not n_jobs:
        return 1
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def _fork_available() -> bool:
    try:
        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


def process_map(fn: Callable, items: Sequence, n_jobs: int | None = 1,
                *, chunksize: int = 1) -> List:
    """``[fn(x) for x in items]`` en paralelo con procesos fork (orden preservado).

    Determinismo: el resultado es idéntico al secuencial siempre que ``fn`` sea
    determinista en su argumento (los solvers/evaluador siembran sus RNG a partir
    de la instancia, no de estado global compartido). Si ``n_jobs<=1``, no hay
    fork en la plataforma, o hay muy pocos items, cae al bucle secuencial.
    """
    items = list(items)
    jobs = min(effective_jobs(n_jobs), len(items)) if items else 1
    if jobs <= 1 or len(items) <= 1 or not _fork_available():
        return [fn(x) for x in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs) as pool:
        return pool.map(fn, items, chunksize=max(1, chunksize))
