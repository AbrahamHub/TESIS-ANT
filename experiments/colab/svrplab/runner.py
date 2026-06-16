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

from . import metrics, stochastic
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


def run_solver(solver, solver_name: str, bank: dict, env, proto: Protocol, *,
               rescore: bool = True, save: bool = True, verbose: bool = True,
               cost_samples: bool = False) -> pd.DataFrame:
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
    """
    paradigm, slug = PARADIGM_OF.get(solver_name, (0, "cross"))
    rows: List[Dict] = []
    samples: Dict[str, np.ndarray] = {}

    pairs = [(s, i, inst) for s in sorted(bank) for i, inst in enumerate(bank[s])]
    for s, i, inst in pairs:
        t0 = time.time()
        sol = solver.solve(inst, num_realizations=proto.realizations)
        wall = time.time() - t0
        if rescore:
            sol, sc = _rescore(sol, inst, proto)
            if cost_samples:
                samples[f"{s}:{i}"] = sc.total_samples
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
        _persist(df, samples, env, slug, solver_name, proto)
    return df


def _persist(df, samples, env, slug, solver_name, proto: Protocol):
    outdir: Path = env.paths.results / slug
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / f"{solver_name}_metrics.csv"
    df.to_csv(csv, index=False)
    meta = outdir / f"{solver_name}_run.json"
    meta.write_text(json.dumps({
        "solver": solver_name,
        "protocol": proto.as_dict(),
        "device": env.device,
        "aggregate": metrics.aggregate_by_size(df).to_dict(orient="records"),
    }, indent=2))
    if samples:
        np.savez(outdir / f"{solver_name}_samples.npz", **samples)
    print(f"[runner] guardado -> {csv}")


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
