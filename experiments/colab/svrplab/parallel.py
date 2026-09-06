"""Paralelismo multi-proceso (fork) para las fases CPU-bound del pipeline.

En Colab la GPU convive con pocas vCPU (T4: 2, L4: 8, A100: 12). Los cuellos de
botella CPU del pipeline —Gurobi, ACO/Tabu heredados y el evaluador CRN (bucle
Python sobre tramos de ruta)— NO escalan con hilos por el GIL; escalan con
**procesos**. Se usa el contexto ``fork`` (Linux/Colab): los workers heredan
``sys.path`` y los módulos ya importados (incluido el repo oficial de SVRPBench)
por copy-on-write, sin re-import ni serialización del entorno.

Anidamiento: ``process_map`` detecta si ya se ejecuta dentro de un worker
(``in_daemon``) y cae a secuencial en vez de intentar crear un Pool hijo, que
Python prohíbe. Sin esa guarda, un solver que paraleliza internamente (p. ej.
``nco_sl`` al generar etiquetas del maestro) revienta con
``AssertionError: daemonic processes are not allowed to have children`` en
cuanto el runner lo ejecuta con ``n_jobs>1``.

Regla de seguridad: los workers ejecutan SOLO numpy/Gurobi/código Python puro.
Nunca tocan CUDA (forkear un proceso con contexto CUDA es seguro mientras el
hijo no llame a la GPU); por eso ``runner.run_solver`` mantiene secuencial la
fase de *solve* de los solvers GPU y paraleliza únicamente la re-puntuación.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, List, Sequence


def available_cpus() -> int:
    """Núcleos **realmente utilizables** por este proceso.

    En Colab ``os.cpu_count()`` reporta los del anfitrión (8 o más) aunque el
    contenedor solo tenga 2 asignados; ``sched_getaffinity`` sí ve el límite del
    cgroup. Sobresuscribir procesos en un T4 de 2 vCPU es más lento que no
    paralelizar.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:              # macOS / Windows
        return max(1, os.cpu_count() or 1)


def effective_jobs(n_jobs: int | None) -> int:
    """Normaliza ``n_jobs``: None/0/1 → 1 (secuencial); -1 → nº de CPUs usables."""
    if not n_jobs:
        return 1
    if n_jobs < 0:
        return available_cpus()
    return int(n_jobs)


def in_daemon() -> bool:
    """¿Estamos dentro de un worker de un Pool? Los procesos daemónicos no
    pueden tener hijos: intentar anidar un Pool lanza
    ``AssertionError: daemonic processes are not allowed to have children``."""
    try:
        return bool(mp.current_process().daemon)
    except Exception:
        return False


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
    # `in_daemon`: si ya corremos dentro de un worker, anidar un Pool aborta la
    # tarea con AssertionError. Cae a secuencial — el paralelismo del nivel
    # superior ya está saturando las CPU, así que no se pierde rendimiento.
    if jobs <= 1 or len(items) <= 1 or not _fork_available() or in_daemon():
        return [fn(x) for x in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs) as pool:
        return pool.map(fn, items, chunksize=max(1, chunksize))
