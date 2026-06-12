"""Genera figuras adicionales para el informe técnico (carpeta ``figures/report/``):

1. ``instance_example``  — naturaleza de una instancia (depósito, clientes, ventanas).
2. ``stochastic_model``  — los 4 vectores estocásticos de SVRPBench.
3. ``tradeoff``          — costo esperado vs factibilidad (burbuja = nº de vehículos).
4. ``runtime``           — tiempo de cómputo por método (escala log).

Uso: ``PYTHONPATH=src python -m svrpx.report_figures``
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io, stochastic
from ._bootstrap import SVRP_ROOT
from time_windows_generator import sample_time_window  # primitiva oficial

REPORT_DIR = SVRP_ROOT / "figures" / "report"
_PARADIGM_COLOR = {
    "exact-bc": "#1f77b4", "exact-bc-tw": "#17becf",
    "aco": "#ff7f0e", "tabu": "#2ca02c",
    "nco-sl": "#d62728", "nco-sl-feas": "#9467bd", "nco-rl": "#8c564b",
}


def fig_instance_example(seed: int = 22345):
    inst = io.generate_instance(20, seed=seed)
    locs = np.asarray(inst.locations, float)
    tw = np.asarray(inst.time_windows, float)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    cust = list(range(1, locs.shape[0]))
    sc = ax.scatter(locs[cust, 0], locs[cust, 1], c=tw[cust, 0], cmap="viridis",
                    s=90, edgecolors="k", linewidths=0.4, zorder=3)
    for i in cust:
        ax.annotate(f"[{int(tw[i,0]//60)}-{int(tw[i,1]//60)}h]", (locs[i, 0], locs[i, 1]),
                    fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.scatter(locs[0, 0], locs[0, 1], marker="s", s=220, c="crimson",
               edgecolors="k", linewidths=1.0, zorder=4, label="Depósito")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Apertura de ventana de tiempo (min)")
    ax.set_title("Instancia TWCVRP de SVRPBench (n=20): clientes agrupados,\n"
                 "ventanas de tiempo heterogéneas, depósito único")
    ax.set_xlabel("x (unidades de mapa, 0–1000)"); ax.set_ylabel("y")
    ax.legend(loc="upper right"); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "instance_example.png", dpi=150)
    plt.close(fig)


def fig_stochastic_model():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    t = np.linspace(0, 1440, 1441)

    # (1) Congestión: factor de tiempo (mezcla gaussiana, picos 8:00 y 17:00)
    tf = [stochastic.time_factor(x) for x in t]
    axes[0, 0].plot(t / 60, tf, color="#1f77b4")
    axes[0, 0].axvline(8, ls="--", c="grey", lw=0.8); axes[0, 0].axvline(17, ls="--", c="grey", lw=0.8)
    axes[0, 0].set_title("(1) Congestión: factor de tiempo (mezcla gaussiana)")
    axes[0, 0].set_xlabel("hora del día"); axes[0, 0].set_ylabel("factor de congestión")

    # (2) Retraso log-normal: densidad off-peak vs pico
    rng = np.random.default_rng(0)
    for lab, tt, col in [("valle (3:00)", 180, "#2ca02c"), ("pico (8:00)", 480, "#d62728")]:
        rush = stochastic.normal_distribution(tt, 480, 90) + stochastic.normal_distribution(tt, 1020, 90)
        mu, sigma = 0.1 * rush, 0.3 + 0.2 * rush
        samples = np.exp(rng.normal(mu, sigma, 20000))
        axes[0, 1].hist(samples, bins=120, range=(0, 5), density=True, alpha=0.6, label=lab, color=col)
    axes[0, 1].set_title("(2) Retraso log-normal (cola pesada)")
    axes[0, 1].set_xlabel("factor multiplicativo de retraso"); axes[0, 1].set_ylabel("densidad"); axes[0, 1].legend()

    # (3) Accidentes: tasa de Poisson (pico a las 21:00)
    rate = [0.05 * stochastic.normal_distribution(x, 1260, 120) for x in t]
    axes[1, 0].plot(t / 60, rate, color="#9467bd")
    axes[1, 0].axvline(21, ls="--", c="grey", lw=0.8)
    axes[1, 0].set_title("(3) Accidentes: tasa de Poisson λ(t)")
    axes[1, 0].set_xlabel("hora del día"); axes[1, 0].set_ylabel("λ (accidentes/viaje)")

    # (4) Ventanas de tiempo: residencial (bimodal) vs comercial
    res = [sample_time_window(0, 0)[0] / 60 for _ in range(5000)]
    com = [sample_time_window(1, 0)[0] / 60 for _ in range(5000)]
    axes[1, 1].hist(res, bins=48, range=(0, 24), alpha=0.6, label="residencial", color="#1f77b4")
    axes[1, 1].hist(com, bins=48, range=(0, 24), alpha=0.6, label="comercial", color="#ff7f0e")
    axes[1, 1].set_title("(4) Ventanas de tiempo (apertura)")
    axes[1, 1].set_xlabel("hora de apertura"); axes[1, 1].set_ylabel("frecuencia"); axes[1, 1].legend()

    fig.suptitle("Los cuatro vectores estocásticos de SVRPBench", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(REPORT_DIR / "stochastic_model.png", dpi=150)
    plt.close(fig)


def fig_tradeoff(size: int = 20):
    df = pd.read_csv(SVRP_ROOT / "results" / "cross" / "comparison_metrics.csv")
    g = df[df["size"] == size].groupby("solver").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for _, r in g.iterrows():
        ax.scatter(r["expected_cost"], r["feasibility"], s=60 + r["n_vehicles"] * 35,
                   color=_PARADIGM_COLOR.get(r["solver"], "grey"), alpha=0.75, edgecolors="k")
        ax.annotate(f"{r['solver']}\n({r['n_vehicles']:.0f} veh.)",
                    (r["expected_cost"], r["feasibility"]), fontsize=8,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Costo esperado E[c]  (menor es mejor →)")
    ax.set_ylabel("Tasa de factibilidad  (↑ mejor)")
    ax.set_title(f"Tradeoff costo–factibilidad–flota (n={size})\n"
                 "tamaño de burbuja = nº de vehículos · esquina superior-izquierda = ideal")
    ax.set_ylim(-0.08, 1.08); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "tradeoff.png", dpi=150)
    plt.close(fig)


def fig_runtime():
    df = pd.read_csv(SVRP_ROOT / "results" / "cross" / "comparison_metrics.csv")
    g = df.groupby(["solver", "size"])["runtime"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    solvers = list(g.index)
    x = np.arange(len(solvers))
    for k, sz in enumerate(g.columns):
        ax.bar(x + k * 0.38, g[sz].clip(lower=1e-4), 0.38, label=f"n={sz}",
               color=["#4c72b0", "#dd8452"][k % 2])
    ax.set_yscale("log")
    ax.set_xticks(x + 0.19); ax.set_xticklabels(solvers, rotation=20, ha="right")
    ax.set_ylabel("Tiempo de cómputo (s, escala log)")
    ax.set_title("Tiempo de producción de la solución por método\n"
                 "(NCO: inferencia; el entrenamiento es un costo único amortizado)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "runtime.png", dpi=150)
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig_instance_example()
    fig_stochastic_model()
    fig_tradeoff()
    fig_runtime()
    print(f"Figuras del informe generadas en {REPORT_DIR}")


if __name__ == "__main__":
    main()
