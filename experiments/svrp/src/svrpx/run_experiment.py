"""Runner CLI de los experimentos preliminares de SVRPBench.

Carga instancias oficiales, ejecuta uno o varios solvers registrados, evalúa cada
solución con el evaluador estocástico compartido (CRN), guarda métricas
(CSV + JSON) y genera las ayudas visuales.

Implementaciones actuales:
  * ``exact-bc``     — Métodos Exactos, CVRP (ignora ventanas; baseline de costo).
  * ``exact-bc-tw``  — Métodos Exactos, CVRPTW (respeta ventanas nominalmente;
                       aísla el efecto de la estocasticidad).

Uso:
    python -m svrpx.run_experiment --solver exact-bc,exact-bc-tw \
        --sizes 10,20,50 --instances 3 --realizations 200 --time-limit 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import io, viz, metrics, paradigms
from ._bootstrap import SVRP_ROOT, get_solver
from . import solvers  # noqa: F401  (registra exact-bc, exact-bc-tw, aco, tabu)


def _build_meta_solver(name, args):
    cls = get_solver(name)
    return cls(default_realizations=args.realizations, alpha=args.alpha,
               late_penalty=args.late_penalty, accident_scale=args.accident_scale)


def _build_solver(name: str, args):
    cls = get_solver(name)
    if name in ("exact-bc", "exact-bc-tw"):
        kw = dict(time_limit=args.time_limit, mip_gap=args.mip_gap,
                  default_realizations=args.realizations, alpha=args.alpha,
                  late_penalty=args.late_penalty, accident_scale=args.accident_scale,
                  threads=args.threads, verbose=args.verbose)
        if name == "exact-bc-tw":
            kw["tw_penalty"] = args.tw_penalty
        return cls(**kw)
    if name in ("aco", "tabu"):
        return _build_meta_solver(name, args)
    return cls()


def run(args) -> pd.DataFrame:
    sizes: List[int] = [int(s) for s in args.sizes.split(",") if s.strip()]
    solver_names: List[str] = [s.strip() for s in args.solver.split(",") if s.strip()]
    pslug = paradigms.paradigm_dir(solver_names)  # p.ej. "01_exact", "02_metaheuristic", "cross"
    figdir = SVRP_ROOT / "figures" / pslug
    resdir = SVRP_ROOT / "results" / pslug
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)
    print(f"Paradigma: {pslug}  ->  results/{pslug}/  figures/{pslug}/")

    instances = io.load_sizes(sizes, args.instances, base_seed=args.seed,
                              capacity_mode=args.capacity_mode)
    rows, per_instance = [], []

    for sname in solver_names:
        solver = _build_solver(sname, args)
        seen = {s: 0 for s in sizes}
        for size, inst in instances:
            idx = seen[size]; seen[size] += 1
            print(f"[{sname}] n={size} instancia {idx} ...", flush=True)
            try:
                sol = solver.solve(inst, num_realizations=args.realizations)
            except Exception as e:  # p.ej. límite de licencia Gurobi en n grande
                print(f"    SALTADA: {str(e)[:120]}", flush=True)
                continue
            row = metrics.row_from_solution(sname, size, idx, sol)
            rows.append(row)
            per_instance.append({
                "solver": sname, "size": size, "instance": idx,
                "routes": sol.routes, **sol.as_metrics(), "extras": dict(sol.extras),
            })
            g = row["gap"]
            print("    det=%.0f gap=%s | E[c]=%.0f E[c+Q]=%.0f CVaR=%.0f feas=%.2f cvr=%.0f robust=%.2f | %.1fs"
                  % (row["det_cost"], f"{100*g:.1f}%" if np.isfinite(g) else "n/a",
                     row["expected_cost"], row["expected_total"], row["cvar"],
                     row["feasibility"], row["cvr"], row["robustness"], row["runtime"]), flush=True)
            if idx == 0:
                viz.make_all(inst, sol, figdir, solver=sname, size=size,
                             instance_idx=idx, alpha=args.alpha)

    df = metrics.to_dataframe(rows)
    tag = "comparison" if len(solver_names) > 1 else solver_names[0]
    csv_path = resdir / f"{tag}_metrics.csv"
    df.to_csv(csv_path, index=False)
    with open(resdir / f"{tag}_per_instance.json", "w") as f:
        json.dump(per_instance, f, indent=2, default=float)

    if len(solver_names) > 1:
        viz.plot_metrics_comparison(df, figdir / f"{tag}_metrics.png")
    else:
        viz.plot_metrics_bars(df, figdir / f"{tag}_metrics.png", solver=solver_names[0])

    _print_summary(df)
    print(f"\nResultados: {csv_path}")
    print(f"Figuras:    {figdir}")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    agg = metrics.aggregate_by_size(df)
    print("\n" + "=" * 86)
    print("RESUMEN  (promedios por solver y tamaño)")
    print("=" * 86)
    cols = ["solver", "size", "det_cost", "expected_cost", "expected_total", "cvar",
            "feasibility", "cvr", "robustness", "gap", "runtime"]
    show = agg[cols].copy()
    show["size"] = show["size"].astype(int)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(show.to_string(index=False))
    print("\nLectura: E[c]=tiempo de viaje esperado · E[c+Q]=con recurso de 2ª etapa · "
          "CVaR=riesgo de cola · feasibility=tasa de factibilidad · gap=brecha MIP.")


def main() -> None:
    p = argparse.ArgumentParser(description="Experimentos preliminares SVRPBench")
    p.add_argument("--solver", default="exact-bc,exact-bc-tw",
                   help="uno o varios separados por coma")
    p.add_argument("--sizes", default="10,20,50")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--realizations", type=int, default=200)
    p.add_argument("--time-limit", type=float, default=120.0, dest="time_limit")
    p.add_argument("--mip-gap", type=float, default=0.0, dest="mip_gap")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--late-penalty", type=float, default=1.0, dest="late_penalty")
    p.add_argument("--tw-penalty", type=float, default=1.0, dest="tw_penalty")
    p.add_argument("--accident-scale", type=float, default=1.0, dest="accident_scale",
                   help="multiplica la tasa de accidentes Poisson (1=oficial; >1 engruesa la cola/CVaR)")
    p.add_argument("--capacity-mode", default="binding", choices=["binding", "official"],
                   dest="capacity_mode")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--verbose", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
