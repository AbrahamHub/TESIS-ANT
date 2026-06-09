"""Ayudas visuales: cómo el método resuelve la ruta + estadísticas.

Cuatro figuras por (solver, tamaño), guardadas en ``figures/``:
  1. ``*_routes_*``       — mapa: depósito (■), clientes coloreados por apertura de
                            ventana de tiempo, rutas por vehículo con orden (flechas).
  2. ``*_convergence_*``  — Branch & Cut: incumbente vs cota inferior vs tiempo; gap sombreado.
  3. ``*_costhist_*``     — histograma del costo total c+Q sobre realizaciones;
                            líneas en media y CVaR_alpha (cola pesada de SVRPBench).
  4. ``*_metrics``        — barras por tamaño: costo esperado, factibilidad, CVaR, runtime.

Las muestras del histograma se recalculan re-puntuando con la misma semilla
(determinista), por lo que no hace falta almacenarlas en el ``Solution``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import stochastic

_ROUTE_COLORS = plt.cm.tab10.colors


def _depot(instance) -> int:
    return int(instance.metadata.get("depot_index", 0))


def plot_routes(instance, solution, path: Path, *, title: str = "") -> None:
    depot = _depot(instance)
    locs = np.asarray(instance.locations, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # Clientes coloreados por apertura de ventana de tiempo.
    cust = [i for i in range(locs.shape[0]) if i != depot]
    if instance.time_windows is not None:
        tw_open = np.asarray(instance.time_windows, dtype=float)[:, 0]
        sc = ax.scatter(locs[cust, 0], locs[cust, 1], c=tw_open[cust],
                        cmap="viridis", s=45, zorder=3, edgecolors="k", linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Apertura de ventana (min)")
    else:
        ax.scatter(locs[cust, 0], locs[cust, 1], s=45, zorder=3,
                   edgecolors="k", linewidths=0.3)

    # Rutas (depósito -> ... -> depósito) con flechas de orden.
    for r_idx, route in enumerate(solution.routes):
        if not route:
            continue
        seq = [depot] + list(route) + [depot]
        color = _ROUTE_COLORS[r_idx % len(_ROUTE_COLORS)]
        for a, b in zip(seq[:-1], seq[1:]):
            ax.annotate("", xy=locs[b], xytext=locs[a],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                        alpha=0.85, shrinkA=4, shrinkB=4), zorder=2)

    # Depósito.
    ax.scatter(locs[depot, 0], locs[depot, 1], marker="s", s=160, c="crimson",
               edgecolors="k", linewidths=0.8, zorder=4, label="Depósito")

    ax.set_title(title or "Ruta a priori")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(solution, path: Path, *, title: str = "") -> None:
    log = solution.extras.get("convergence_log", [])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _INF = 1e30  # Gurobi usa 1e100 como +/-infinito antes de tener incumbente/cota
    pts = [(float(t), float(b), float(d)) for (t, b, d) in log]
    t = np.array([p[0] for p in pts])
    bst = np.array([p[1] for p in pts])
    bnd = np.array([p[2] for p in pts])
    fin_bst = np.abs(bst) < _INF
    fin_bnd = np.abs(bnd) < _INF
    if pts and fin_bst.any():
        # Cota inferior (donde es finita).
        ax.step(t[fin_bnd], bnd[fin_bnd], where="post", color="navy", lw=1.8,
                label="Cota inferior")
        # Incumbente (sólo tras la primera solución factible).
        tb, vb = t[fin_bst], bst[fin_bst]
        ax.step(tb, vb, where="post", color="crimson", lw=1.8,
                marker="o", ms=3, label="Incumbente (mejor solución)")
        # Brecha sombreada entre cota e incumbente sobre el rango común.
        if fin_bnd.any():
            t0 = max(tb.min(), t[fin_bnd].min())
            grid = t[(t >= t0)]
            if grid.size:
                bst_i = np.interp(grid, tb, vb)
                bnd_i = np.interp(grid, t[fin_bnd], bnd[fin_bnd])
                ax.fill_between(grid, bnd_i, bst_i, color="gold", alpha=0.35,
                                step="post", label="Brecha (gap)")
        lo = bnd[fin_bnd].min() if fin_bnd.any() else vb.min()
        hi = vb.max()
        margin = 0.05 * (hi - lo + 1)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("Tiempo de cómputo (s)")
    else:
        ax.text(0.5, 0.5, "Sin registro de convergencia", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("Costo determinista (distancia)")
    gap = solution.extras.get("gap", float("nan"))
    ax.set_title(title or f"Convergencia Branch & Cut (gap final = {gap:.3%})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cost_hist(instance, solution, path: Path, *, alpha: float = 0.95,
                   seed: Optional[int] = None, title: str = "") -> None:
    if seed is None:
        seed = int(instance.metadata.get("seed", 0))
    R = int(solution.extras.get("realizations", 200))
    late = 1.0
    score = stochastic.score_routes(instance, solution.routes, num_realizations=R,
                                    seed=seed, alpha=alpha, late_penalty=late,
                                    depot=_depot(instance))
    samples = np.asarray(score.total_samples)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(samples, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(score.expected_total, color="black", lw=1.8,
               label=f"Media c+Q = {score.expected_total:.0f}")
    ax.axvline(score.cvar, color="crimson", lw=1.8, ls="--",
               label=f"CVaR$_{{{alpha:.2f}}}$ = {score.cvar:.0f}")
    ax.set_xlabel("Costo total por realización (c + Q)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title or "Distribución de costo bajo estocasticidad SVRPBench")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metrics_bars(df, path: Path, *, solver: str = "exact-bc") -> None:
    sub = df[df["solver"] == solver].groupby("size").mean(numeric_only=True).reset_index()
    sizes = sub["size"].astype(int).astype(str).tolist()
    panels = [
        ("expected_cost", "Costo esperado E[c]", "steelblue"),
        ("feasibility", "Tasa de factibilidad", "seagreen"),
        ("cvar", "CVaR (c+Q)", "crimson"),
        ("runtime", "Runtime (s)", "darkorange"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (col, label, color) in zip(axes, panels):
        ax.bar(sizes, sub[col].values, color=color, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Clientes")
        if col == "feasibility":
            ax.set_ylim(0, 1)
    fig.suptitle(f"Métricas por tamaño — {solver}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metrics_comparison(df, path: Path) -> None:
    """Barras agrupadas por tamaño comparando varios solvers."""
    solvers = list(df["solver"].unique())
    sizes = sorted(df["size"].unique())
    panels = [
        ("expected_cost", "Costo esperado E[c]"),
        ("feasibility", "Tasa de factibilidad"),
        ("cvar", "CVaR (c+Q)"),
        ("expected_total", "Costo total E[c+Q]"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    width = 0.8 / max(1, len(solvers))
    xpos = np.arange(len(sizes))
    for ax, (col, label) in zip(axes, panels):
        for s_idx, sv in enumerate(solvers):
            sub = df[df["solver"] == sv].groupby("size")[col].mean()
            vals = [sub.get(sz, np.nan) for sz in sizes]
            ax.bar(xpos + s_idx * width, vals, width, label=sv,
                   color=_ROUTE_COLORS[s_idx % len(_ROUTE_COLORS)], alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(xpos + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("Clientes")
        if col == "feasibility":
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Comparación de baselines exactos", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_all(instance, solution, figdir: Path, *, solver: str, size: int,
             instance_idx: int, alpha: float = 0.95) -> List[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"{solver}_n{size}_i{instance_idx}"
    paths = []
    p1 = figdir / f"{tag}_routes.png"
    plot_routes(instance, solution, p1,
                title=f"{solver} · n={size} · {solution.extras.get('n_routes', 0)} rutas "
                      f"· det={solution.extras.get('det_cost', float('nan')):.0f}")
    p2 = figdir / f"{tag}_convergence.png"
    plot_convergence(solution, p2, title=f"Branch & Cut · n={size} · "
                     f"gap={solution.extras.get('gap', float('nan')):.2%}")
    p3 = figdir / f"{tag}_costhist.png"
    plot_cost_hist(instance, solution, p3, alpha=alpha,
                   title=f"Costo estocástico · n={size} · feas={solution.feasibility:.2f}")
    paths.extend([p1, p2, p3])
    return paths
