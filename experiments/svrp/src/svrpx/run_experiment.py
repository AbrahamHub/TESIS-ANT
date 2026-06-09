"""Runner CLI de los experimentos preliminares de SVRPBench.

Carga instancias oficiales, ejecuta un solver registrado (implementación 1/5:
``exact-bc``), evalúa cada solución con el evaluador estocástico compartido,
guarda métricas (CSV + JSON) y genera las ayudas visuales.

Uso:
    python -m svrpx.run_experiment --solver exact-bc --sizes 10,20,50 \
        --instances 3 --realizations 200 --time-limit 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import io, viz, metrics
from ._bootstrap import SVRP_ROOT, get_solver
from . import solvers  # noqa: F401  (registra exact-bc)


def _build_solver(name: str, args):
    cls = get_solver(name)
    if name == "exact-bc":
        return cls(time_limit=args.time_limit, mip_gap=args.mip_gap,
                   default_realizations=args.realizations, alpha=args.alpha,
                   late_penalty=args.late_penalty, threads=args.threads,
                   verbose=args.verbose)
    return cls()


def run(args) -> pd.DataFrame:
    sizes: List[int] = [int(s) for s in args.sizes.split(",") if s.strip()]
    figdir = SVRP_ROOT / "figures"
    resdir = SVRP_ROOT / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    solver = _build_solver(args.solver, args)
    rows, per_instance = [], []

    # Contador de instancia por tamaño para nombrar figuras.
    seen = {s: 0 for s in sizes}
    for size, inst in io.load_sizes(sizes, args.instances, base_seed=args.seed):
        idx = seen[size]; seen[size] += 1
        print(f"[{args.solver}] n={size} instancia {idx} ...", flush=True)
        sol = solver.solve(inst, num_realizations=args.realizations)
        row = metrics.row_from_solution(args.solver, size, idx, sol)
        rows.append(row)
        per_instance.append({
            "solver": args.solver, "size": size, "instance": idx,
            "routes": sol.routes, **sol.as_metrics(),
            "extras": dict(sol.extras),
        })
        print("    det_cost=%.0f gap=%.2f%% | E[c]=%.0f E[c+Q]=%.0f CVaR=%.0f "
              "feas=%.2f cvr=%.0f robust=%.2f | %.1fs"
              % (row["det_cost"], 100 * (row["gap"] if np.isfinite(row["gap"]) else float("nan")),
                 row["expected_cost"], row["expected_total"], row["cvar"],
                 row["feasibility"], row["cvr"], row["robustness"], row["runtime"]), flush=True)

        if idx == 0:  # figuras para la instancia representativa de cada tamaño
            viz.make_all(inst, sol, figdir, solver=args.solver, size=size,
                         instance_idx=idx, alpha=args.alpha)

    df = metrics.to_dataframe(rows)
    csv_path = resdir / f"{args.solver}_metrics.csv"
    df.to_csv(csv_path, index=False)
    with open(resdir / f"{args.solver}_per_instance.json", "w") as f:
        json.dump(per_instance, f, indent=2, default=float)
    viz.plot_metrics_bars(df, figdir / f"{args.solver}_metrics.png", solver=args.solver)

    _print_summary(df, args.solver)
    print(f"\nResultados: {csv_path}")
    print(f"Figuras:    {figdir}")
    return df


def _print_summary(df: pd.DataFrame, solver: str) -> None:
    agg = metrics.aggregate_by_size(df)
    print("\n" + "=" * 78)
    print(f"RESUMEN — {solver}  (promedios por tamaño)")
    print("=" * 78)
    cols = ["size", "det_cost", "expected_cost", "expected_total", "cvar",
            "feasibility", "cvr", "robustness", "gap", "runtime"]
    show = agg[cols].copy()
    show["size"] = show["size"].astype(int)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(show.to_string(index=False))
    print("\nLectura: E[c] >= det_cost (la estocasticidad degrada el costo) y "
          "CVaR >= E[c+Q] (cola pesada).")


def main() -> None:
    p = argparse.ArgumentParser(description="Experimentos preliminares SVRPBench")
    p.add_argument("--solver", default="exact-bc")
    p.add_argument("--sizes", default="10,20,50")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--realizations", type=int, default=200)
    p.add_argument("--time-limit", type=float, default=120.0, dest="time_limit")
    p.add_argument("--mip-gap", type=float, default=0.0, dest="mip_gap")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--late-penalty", type=float, default=1.0, dest="late_penalty")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--verbose", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
