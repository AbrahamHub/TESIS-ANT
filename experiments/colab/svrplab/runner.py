"""Orquestador común: corre un solver sobre el banco canónico, re-puntúa con el
evaluador compartido bajo el protocolo homologado, y persiste métricas + solución.

Todos los notebooks llaman a ``run_solver(...)`` para garantizar exactamente el mismo
trato: mismas instancias, mismos escenarios ξ (CRN por seed de instancia), misma
re-puntuación. Lo único que cambia entre paradigmas es **cómo** el solver produce las
rutas; la evaluación es idéntica.

Robustez para Colab (sesiones que se desconectan, GPU que se agota, instancias que
revientan):

  * **Reanudación** (``resume=True``): cada instancia terminada se anota en un
    ``*_checkpoint.jsonl``. Al relanzar la celda, las instancias ya resueltas se
    saltan. El checkpoint guarda la huella del banco y del protocolo: si cambias
    ``SIZES``/``N_INSTANCES``/protocolo, se invalida solo y se recalcula todo.
  * **Aislamiento de errores** (``on_error="record"``): una instancia que falla no
    tumba la corrida; se anota con métricas NaN y ``error`` en el CSV, y el resto
    continúa. Al final se imprime un resumen de fallos.
  * **Checkpoint incremental**: se persiste cada ``checkpoint_every`` instancias, no
    solo al final. Una desconexión pierde como mucho ese bloque.
  * **Presupuesto de tiempo** (``time_budget_s``): corta limpiamente y deja
    persistido lo hecho, con aviso de qué falta.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as svrp_data, metrics, stochastic
from .parallel import effective_jobs, process_map
from .protocol import PARADIGM_OF, Protocol


# --------------------------------------------------------------------------- #
# Re-puntuación homologada
# --------------------------------------------------------------------------- #


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
    ``solve``, así que el resultado es idéntico al secuencial.

    Devuelve ``(sol, sc, wall, err)``; ``err`` no es None si la instancia falló, de
    modo que un fallo viaja de vuelta como dato en vez de matar el pool entero.
    """
    solver, inst, proto, rescore = args
    t0 = time.time()
    try:
        sol = solver.solve(inst, num_realizations=proto.realizations)
        sc = None
        if rescore:
            sol, sc = _rescore(sol, inst, proto)
        return sol, sc, time.time() - t0, None
    except Exception:
        return None, None, time.time() - t0, traceback.format_exc(limit=6)


def _rescore_task(args):
    inst, sol, proto = args
    try:
        return (*_rescore(sol, inst, proto), None)
    except Exception:
        return sol, None, traceback.format_exc(limit=6)


# --------------------------------------------------------------------------- #
# Checkpointing / reanudación
# --------------------------------------------------------------------------- #


def _run_signature(bank, proto: Protocol, solver_name: str) -> str:
    """Huella de (banco + protocolo + solver). Si cambia, el checkpoint previo deja
    de ser válido y la corrida se reinicia sola: nunca se mezclan resultados de
    protocolos distintos en el mismo CSV."""
    import hashlib
    payload = json.dumps({
        "solver": solver_name,
        "bank": svrp_data.bank_fingerprint(bank),
        "sizes": sorted(int(s) for s in bank),
        "n": {int(s): len(v) for s, v in bank.items()},
        "protocol": proto.as_dict(),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class _Checkpoint:
    """Bitácora append-only de instancias terminadas (una línea JSON por instancia)."""

    def __init__(self, path: Path, signature: str):
        self.path = path
        self.signature = signature
        self.rows: Dict[str, dict] = {}
        self.routes: Dict[str, list] = {}

    def load(self) -> int:
        if not self.path.exists():
            return 0
        kept, stale = 0, False
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("signature") != self.signature:
                stale = True
                break
            key = rec["key"]
            self.rows[key] = rec["row"]
            self.routes[key] = rec.get("routes", [])
            kept += 1
        if stale:
            # El banco o el protocolo cambiaron: descartar y empezar limpio.
            self.rows.clear(); self.routes.clear()
            self.path.unlink(missing_ok=True)
            return 0
        return kept

    def append(self, key: str, row: dict, routes: list) -> None:
        self.rows[key] = row
        self.routes[key] = routes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"signature": self.signature, "key": key,
                                 "row": row, "routes": routes},
                                default=_json_safe) + "\n")

    def done(self) -> set:
        return set(self.rows)


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _failure_row(solver_name, paradigm, size, idx, seed, err, wall):
    """Fila con métricas NaN para una instancia que falló: la corrida sigue y el
    fallo queda **en el CSV**, no solo en la consola."""
    row = {c: np.nan for c in metrics.METRIC_COLUMNS}
    row.update({"solver": solver_name, "paradigm": paradigm, "size": size,
                "instance": idx, "seed": seed, "runtime": wall})
    row["error"] = (err or "").strip().splitlines()[-1][:300] if err else "error"
    return row


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #


def run_solver(solver, solver_name: str, bank: dict, env, proto: Protocol, *,
               rescore: bool = True, save: bool = True, verbose: bool = True,
               cost_samples: bool = False, n_jobs: int = 1,
               resume: bool = True, on_error: str = "record",
               time_budget_s: Optional[float] = None,
               checkpoint_every: int = 1) -> pd.DataFrame:
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
    resume : reanudar desde el checkpoint si el banco y el protocolo coinciden.
    on_error : ``"record"`` anota el fallo y sigue; ``"raise"`` propaga la excepción.
    time_budget_s : corta limpiamente al superar este tiempo (segundos), dejando
        persistido lo ya resuelto. ``None`` = sin límite.
    checkpoint_every : cada cuántas instancias se escribe el checkpoint.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error debe ser 'record' o 'raise'")

    paradigm, slug = PARADIGM_OF.get(solver_name, (0, "cross"))
    outdir: Path = env.paths.results / slug
    sig = _run_signature(bank, proto, solver_name)
    ck = _Checkpoint(outdir / f"{solver_name}_checkpoint.jsonl", sig)
    n_resumed = ck.load() if resume else 0

    pairs = [(s, i, inst) for s in sorted(bank) for i, inst in enumerate(bank[s])]
    todo = [(s, i, inst) for (s, i, inst) in pairs if f"{s}:{i}" not in ck.done()]

    if verbose:
        total = len(pairs)
        print(f"[runner] {solver_name}: {total} instancias | "
              f"reanudadas {n_resumed} | pendientes {len(todo)} | firma {sig}")
        print(f"[runner] protocolo: R_eval={proto.realizations} r∈[0,{proto.realizations}) | "
              f"búsqueda R={proto.search_realizations} r∈[{proto.search_offset},"
              f"{proto.search_offset + proto.search_realizations}) | "
              f"disjunta={proto.disjoint_search}")

    samples: Dict[str, np.ndarray] = {}
    failures: List[tuple] = []
    jobs = effective_jobs(n_jobs)
    gpu_solver = getattr(solver, "device", "cpu") == "cuda"
    t_start = time.time()
    stopped_early = False

    # Se procesa por bloques para poder hacer checkpoint y respetar el presupuesto
    # de tiempo aunque el pool paralelo trabaje en lote.
    block = max(1, int(checkpoint_every)) if not (jobs > 1 and not gpu_solver) else \
        max(int(checkpoint_every), jobs)

    for lo in range(0, len(todo), block):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            stopped_early = True
            break
        chunk = todo[lo:lo + block]

        if jobs > 1 and not gpu_solver:
            results = process_map(
                _solve_task, [(solver, inst, proto, rescore) for _, _, inst in chunk],
                n_jobs=jobs)
        else:
            # Solve secuencial (obligatorio para GPU); re-puntuación en paralelo aparte.
            solved = []
            for s, i, inst in chunk:
                t0 = time.time()
                try:
                    sol = solver.solve(inst, num_realizations=proto.realizations)
                    solved.append((sol, time.time() - t0, None))
                except Exception:
                    if on_error == "raise":
                        raise
                    solved.append((None, time.time() - t0, traceback.format_exc(limit=6)))
            if rescore and jobs > 1:
                ok = [(inst, sol) for (_, _, inst), (sol, _, e) in zip(chunk, solved)
                      if sol is not None and e is None]
                scored = process_map(_rescore_task,
                                     [(inst, sol, proto) for inst, sol in ok],
                                     n_jobs=jobs) if ok else []
                it = iter(scored)
                results = []
                for (sol, wall, err) in solved:
                    if sol is None or err is not None:
                        results.append((None, None, wall, err))
                    else:
                        rsol, rsc, rerr = next(it)
                        results.append((rsol, rsc, wall, rerr))
            elif rescore:
                results = []
                for (_, _, inst), (sol, wall, err) in zip(chunk, solved):
                    if sol is None or err is not None:
                        results.append((None, None, wall, err))
                        continue
                    try:
                        rsol, rsc = _rescore(sol, inst, proto)
                        results.append((rsol, rsc, wall, None))
                    except Exception:
                        if on_error == "raise":
                            raise
                        results.append((None, None, wall, traceback.format_exc(limit=6)))
            else:
                results = [(sol, None, wall, err) for sol, wall, err in solved]

        for (s, i, inst), (sol, sc, wall, err) in zip(chunk, results):
            key = f"{s}:{i}"
            seed = int(inst.metadata.get("seed", 0))
            if err is not None or sol is None:
                if on_error == "raise":
                    raise RuntimeError(f"{solver_name} falló en n={s} inst={i}:\n{err}")
                failures.append((s, i, err))
                row = _failure_row(solver_name, paradigm, s, i, seed, err, wall)
                ck.append(key, row, [])
                if verbose:
                    print(f"  [{solver_name}] n={s} inst={i}: FALLO -> "
                          f"{row['error']} (se anota y continúa)")
                continue
            if cost_samples and sc is not None:
                samples[key] = sc.total_samples
            row = metrics.row_from_solution(solver_name, paradigm, s, i, seed, sol)
            row["error"] = ""
            ck.append(key, row, [list(map(int, r)) for r in sol.routes])
            if verbose:
                print(f"  [{solver_name}] n={s} inst={i}: E[c]={row['expected_cost']:.1f} "
                      f"E[c+Q]={row['expected_total']:.1f} CVaR={row['cvar']:.1f} "
                      f"feas={row['feasibility']:.2f} veh={row['n_vehicles']} "
                      f"t={row['runtime']:.3f}s (wall {wall:.1f}s)")

    rows = [ck.rows[k] for k in sorted(ck.rows, key=_key_order)]
    routes_map = {k: ck.routes.get(k, []) for k in sorted(ck.rows, key=_key_order)}
    df = metrics.to_dataframe(rows)

    if verbose:
        _report_run(solver_name, df, failures, stopped_early, len(pairs),
                    time.time() - t_start)
    if save:
        _persist(df, samples, routes_map, bank, env, slug, solver_name, proto,
                 failures=failures, complete=(len(ck.rows) == len(pairs)
                                              and not stopped_early))
    return df


def _key_order(k: str):
    s, i = k.split(":")
    return (int(s), int(i))


def _report_run(solver_name, df, failures, stopped_early, n_total, elapsed):
    ok = int((df.get("error", pd.Series([""] * len(df))).fillna("") == "").sum()) \
        if len(df) else 0
    print(f"[runner] {solver_name}: {ok}/{n_total} instancias con métricas válidas "
          f"en {elapsed:.1f}s")
    if failures:
        print(f"[runner] {len(failures)} FALLOS (anotados en el CSV con error!=''):")
        for s, i, err in failures[:5]:
            last = (err or "").strip().splitlines()[-1][:160]
            print(f"         n={s} inst={i}: {last}")
        if len(failures) > 5:
            print(f"         ... y {len(failures) - 5} más")
        print("[runner] Relanza la celda: la reanudación reintenta SOLO lo que falta "
              "si borras sus líneas del *_checkpoint.jsonl, o corrige la causa y "
              "borra el checkpoint para recalcular.")
    if stopped_early:
        print("[runner] ATENCIÓN: se agotó time_budget_s. Lo resuelto quedó "
              "persistido; vuelve a ejecutar la celda para continuar donde iba.")


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #


def _persist(df, samples, routes_map, bank, env, slug, solver_name, proto: Protocol,
             *, failures=None, complete=True):
    outdir: Path = env.paths.results / slug
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / f"{solver_name}_metrics.csv"
    df.to_csv(csv, index=False)
    meta = outdir / f"{solver_name}_run.json"
    meta.write_text(json.dumps({
        "solver": solver_name,
        "protocol": proto.as_dict(),
        # Auditoría anti-fuga: rangos de ξ efectivamente usados.
        "scenario_ranges": {
            "eval": [0, int(proto.realizations)],
            "search": [int(proto.search_offset),
                       int(proto.search_offset + proto.search_realizations)],
            "disjoint": bool(proto.disjoint_search),
        },
        "device": env.device,
        "bank_fingerprint": svrp_data.bank_fingerprint(bank),
        "size_fingerprints": svrp_data.size_fingerprints(bank),
        "sizes": sorted(int(s) for s in bank),
        "n_instances": {int(s): len(v) for s, v in bank.items()},
        # complete = se intentaron TODAS las instancias (no se cortó por tiempo).
        # all_valid = además ninguna falló. El notebook 06 audita `all_valid`.
        "complete": bool(complete),
        "all_valid": bool(complete and not (failures or [])),
        "n_failures": len(failures or []),
        "failures": [{"size": s, "instance": i,
                      "error": (e or "").strip().splitlines()[-1][:300]}
                     for s, i, e in (failures or [])],
        "aggregate": metrics.aggregate_by_size(df).to_dict(orient="records"),
    }, indent=2, default=_json_safe))
    # Rutas por instancia: habilitan el estudio de caso comparativo del
    # notebook 06 (misma instancia dibujada método a método) y re-evaluaciones.
    (outdir / f"{solver_name}_routes.json").write_text(json.dumps(routes_map))
    if samples:
        np.savez(outdir / f"{solver_name}_samples.npz", **samples)
    flag = "" if complete else "  [PARCIAL]"
    print(f"[runner] guardado -> {csv} (+rutas, +run.json){flag}")


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #


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


def load_all_results(env, *, drop_failed: bool = True) -> pd.DataFrame:
    """Carga y concatena todos los ``*_metrics.csv`` bajo ``results/`` (para el
    notebook de comparación final). Cada paradigma escribió en su carpeta.

    ``drop_failed`` descarta las filas de instancias que fallaron (``error`` no
    vacío) para que no contaminen medias ni pruebas estadísticas; el conteo se
    informa por pantalla.
    """
    frames = []
    for csv in sorted(env.paths.results.rglob("*_metrics.csv")):
        try:
            frames.append(pd.read_csv(csv))
        except Exception as e:
            print(f"[runner] omito {csv}: {e}")
    if not frames:
        return pd.DataFrame(columns=metrics.METRIC_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if drop_failed and "error" in df.columns:
        bad = df["error"].fillna("").astype(str).str.len() > 0
        if bad.any():
            print(f"[runner] descarto {int(bad.sum())} filas con error "
                  f"({', '.join(sorted(df.loc[bad, 'solver'].unique()))})")
            df = df.loc[~bad].reset_index(drop=True)
    return df


def clear_checkpoint(env, solver_name: str) -> int:
    """Borra el checkpoint de un solver para forzar el recálculo completo.
    Úsalo tras corregir la causa de un fallo. Devuelve cuántos archivos borró."""
    n = 0
    for p in sorted(env.paths.results.rglob(f"{solver_name}_checkpoint.jsonl")):
        p.unlink()
        n += 1
        print(f"[runner] checkpoint borrado: {p}")
    if n == 0:
        print(f"[runner] no había checkpoint de {solver_name}")
    return n
