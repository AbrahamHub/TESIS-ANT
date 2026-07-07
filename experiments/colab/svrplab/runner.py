"""Orquestador común: corre un solver sobre el banco canónico, re-puntúa con el
evaluador compartido bajo el protocolo homologado, y persiste métricas + solución.

Todos los notebooks llaman a ``run_solver(...)`` para garantizar exactamente el mismo
trato: mismas instancias, mismos escenarios ξ (CRN por seed de instancia), misma
re-puntuación. Lo único que cambia entre paradigmas es **cómo** el solver produce las
rutas; la evaluación es idéntica.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as svrp_data, metrics, stochastic
from .parallel import effective_jobs, process_map
from .protocol import PARADIGM_OF, Protocol


def _rescore(sol, inst, proto: Protocol):
    """Re-puntúa las rutas del solver con el evaluador compartido (homologación).
    Conserva del solver solo lo propio de su mecánica (``runtime`` y extras como
    ``det_cost``/``gap``/``train_time_s``)."""
    depot = int(inst.metadata.get("depot_index", 0))
    sc = stochastic.score_routes(
        inst, sol.routes, seed=int(inst.metadata.get("seed", 0)), depot=depot,
        **proto.eval_kwargs())
    extras = dict(sol.extras or {})
    extras.update(sc.as_extras())
    extras["n_routes"] = sum(1 for r in sol.routes if len(r) > 0)
    sol.total_cost = sc.expected_cost
    sol.feasibility = sc.feasibility
    sol.cvr = sc.cvr
    sol.waiting_time = sc.waiting_time
    sol.robustness = sc.robustness
    sol.extras = extras
    return sol, sc


def _solve_task(args):
    """Tarea por instancia para el pool fork (solvers CPU): solve + re-puntuación en
    el worker. Los solvers CPU siembran sus RNG desde la instancia dentro de
    ``solve``, así que el resultado es idéntico al secuencial."""
    solver, inst, proto, rescore = args
    t0 = time.time()
    sol = solver.solve(inst, num_realizations=proto.realizations)
    wall = time.time() - t0
    sc = None
    if rescore:
        sol, sc = _rescore(sol, inst, proto)
    return sol, sc, wall


def _rescore_task(args):
    inst, sol, proto = args
    return _rescore(sol, inst, proto)


def run_solver(solver, solver_name: str, bank: dict, env, proto: Protocol, *,
               rescore: bool = True, save: bool = True, verbose: bool = True,
               cost_samples: bool = False, n_jobs: int = 1) -> pd.DataFrame:
    """Ejecuta ``solver`` (instancia ya construida) sobre todo el ``bank``.

    Parámetros
    ----------
    solver : objeto con ``.solve(instance, num_realizations=...) -> Solution``.
    solver_name : nombre canónico (define paradigma/carpeta de salida).
    bank : ``{size: [Instance, ...]}`` del banco canónico.
    env : ``bootstrap.Env`` (rutas, device).
    proto : ``Protocol`` con las condiciones homologadas.
    rescore : re-puntuar con el evaluador compartido (recomendado siempre True).
    cost_samples : si True, guarda las muestras de costo por instancia (para CVaR/plots).
    n_jobs : procesos paralelos (fork) para las fases CPU. Con solvers CPU
        (exact/aco/tabu) paraleliza solve+re-puntuación por instancia; con solvers
        GPU (``solver.device == 'cuda'``) el solve queda secuencial (CUDA no es
        fork-safe) y solo se paraleliza la re-puntuación CRN. -1 = nº de CPUs.
    """
    paradigm, slug = PARADIGM_OF.get(solver_name, (0, "cross"))
    rows: List[Dict] = []
    samples: Dict[str, np.ndarray] = {}
    jobs = effective_jobs(n_jobs)
    gpu_solver = getattr(solver, "device", "cpu") == "cuda"

    pairs = [(s, i, inst) for s in sorted(bank) for i, inst in enumerate(bank[s])]

    if jobs > 1 and not gpu_solver:
        results = process_map(_solve_task,
                              [(solver, inst, proto, rescore) for _, _, inst in pairs],
                              n_jobs=jobs)
    else:
        # Solve secuencial (obligatorio para GPU); re-puntuación en paralelo aparte.
        solved = []
        for s, i, inst in pairs:
            t0 = time.time()
            sol = solver.solve(inst, num_realizations=proto.realizations)
            solved.append((sol, time.time() - t0))
        if rescore and jobs > 1:
            scored = process_map(_rescore_task,
                                 [(inst, sol, proto) for (_, _, inst), (sol, _) in zip(pairs, solved)],
                                 n_jobs=jobs)
            results = [(sol_sc[0], sol_sc[1], wall)
                       for sol_sc, (_, wall) in zip(scored, solved)]
        elif rescore:
            results = [(*_rescore(sol, inst, proto), wall)
                       for (_, _, inst), (sol, wall) in zip(pairs, solved)]
        else:
            results = [(sol, None, wall) for sol, wall in solved]

    routes_map: Dict[str, List] = {}
    for (s, i, inst), (sol, sc, wall) in zip(pairs, results):
        if cost_samples and sc is not None:
            samples[f"{s}:{i}"] = sc.total_samples
        routes_map[f"{s}:{i}"] = [list(map(int, r)) for r in sol.routes]
        seed = int(inst.metadata.get("seed", 0))
        row = metrics.row_from_solution(solver_name, paradigm, s, i, seed, sol)
        rows.append(row)
        if verbose:
            print(f"  [{solver_name}] n={s} inst={i}: E[c]={row['expected_cost']:.1f} "
                  f"E[c+Q]={row['expected_total']:.1f} CVaR={row['cvar']:.1f} "
                  f"feas={row['feasibility']:.2f} veh={row['n_vehicles']} "
                  f"t={row['runtime']:.3f}s (wall {wall:.1f}s)")

    df = metrics.to_dataframe(rows)
    if save:
        _persist(df, samples, routes_map, bank, env, slug, solver_name, proto)
    return df


def _persist(df, samples, routes_map, bank, env, slug, solver_name, proto: Protocol):
    outdir: Path = env.paths.results / slug
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / f"{solver_name}_metrics.csv"
    df.to_csv(csv, index=False)
    meta = outdir / f"{solver_name}_run.json"
    meta.write_text(json.dumps({
        "solver": solver_name,
        "protocol": proto.as_dict(),
        "device": env.device,
        "bank_fingerprint": svrp_data.bank_fingerprint(bank),
        "size_fingerprints": svrp_data.size_fingerprints(bank),
        "sizes": sorted(int(s) for s in bank),
        "n_instances": {int(s): len(v) for s, v in bank.items()},
        "aggregate": metrics.aggregate_by_size(df).to_dict(orient="records"),
    }, indent=2))
    # Rutas por instancia: habilitan el estudio de caso comparativo del
    # notebook 06 (misma instancia dibujada método a método) y re-evaluaciones.
    (outdir / f"{solver_name}_routes.json").write_text(json.dumps(routes_map))
    if samples:
        np.savez(outdir / f"{solver_name}_samples.npz", **samples)
    print(f"[runner] guardado -> {csv} (+rutas, +run.json)")


def load_routes(env, solver_name: str) -> Dict[str, List]:
    """Carga las rutas persistidas de un solver (``{"size:instance": rutas}``)."""
    for p in sorted(env.paths.results.rglob(f"{solver_name}_routes.json")):
        return json.loads(p.read_text())
    return {}


def load_run_meta(env) -> Dict[str, Dict]:
    """Carga los ``*_run.json`` de todos los solvers (para auditar en el
    notebook 06 que todos usaron el MISMO banco y protocolo: piso parejo)."""
    out = {}
    for p in sorted(env.paths.results.rglob("*_run.json")):
        try:
            meta = json.loads(p.read_text())
            out[meta.get("solver", p.stem)] = meta
        except Exception as e:
            print(f"[runner] omito {p}: {e}")
    return out


def load_samples(env, solver_name: str) -> Dict[str, np.ndarray]:
    """Carga las muestras de costo total c+Q persistidas (``cost_samples=True``)."""
    for p in sorted(env.paths.results.rglob(f"{solver_name}_samples.npz")):
        raw = np.load(p)
        return {k: raw[k] for k in raw.files}
    return {}


def load_all_results(env) -> pd.DataFrame:
    """Carga y concatena todos los ``*_metrics.csv`` bajo ``results/`` (para el
    notebook de comparación final). Cada paradigma escribió en su carpeta."""
    frames = []
    for csv in sorted(env.paths.results.rglob("*_metrics.csv")):
        try:
            frames.append(pd.read_csv(csv))
        except Exception as e:
            print(f"[runner] omito {csv}: {e}")
    if not frames:
        return pd.DataFrame(columns=metrics.METRIC_COLUMNS)
    return pd.concat(frames, ignore_index=True)
