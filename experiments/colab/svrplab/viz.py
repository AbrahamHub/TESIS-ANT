"""Figuras del pipeline (matplotlib). Todas guardan en ``env.paths.figures/<slug>/``.

Reportadas por instancia/solver: ruta, histograma de costo + CVaR, convergencia (B&C)
y curva de entrenamiento (NCO/EHBG). Comparativas entre paradigmas: barras por métrica,
tradeoff costo–factibilidad–flota y tiempo de inferencia.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _ensure(env, slug: str) -> Path:
    d = env.paths.figures / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_instance(inst, *, title: str = "", save: Optional[Path] = None):
    import matplotlib.pyplot as plt
    locs = np.asarray(inst.locations)
    depot = int(inst.metadata.get("depot_index", 0))
    tw = np.asarray(inst.time_windows) if inst.time_windows is not None else None
    fig, ax = plt.subplots(figsize=(6, 5))
    cust = [i for i in range(len(locs)) if i != depot]
    c = (tw[cust, 0] if tw is not None else None)
    sc = ax.scatter(locs[cust, 0], locs[cust, 1], c=c, cmap="viridis", s=40, zorder=3)
    ax.scatter([locs[depot, 0]], [locs[depot, 1]], marker="s", c="red", s=120,
               label="Depósito", zorder=4)
    if tw is not None:
        plt.colorbar(sc, ax=ax, label="Apertura de ventana (min)")
    ax.set_title(title or f"Instancia TWCVRP n={len(locs)-1}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_routes(inst, routes: List[List[int]], *, title: str = "", save: Optional[Path] = None):
    import matplotlib.pyplot as plt
    locs = np.asarray(inst.locations)
    depot = int(inst.metadata.get("depot_index", 0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(locs[1:, 0], locs[1:, 1], c="0.5", s=25, zorder=2)
    ax.scatter([locs[depot, 0]], [locs[depot, 1]], marker="s", c="red", s=120, zorder=4)
    cmap = plt.get_cmap("tab10")
    for k, r in enumerate(routes):
        if not r:
            continue
        path = [depot] + list(r) + [depot]
        ax.plot(locs[path, 0], locs[path, 1], "-o", ms=3, color=cmap(k % 10),
                lw=1.2, zorder=3, label=f"Ruta {k+1}")
    ax.set_title(title or f"Rutas ({sum(1 for r in routes if r)} vehículos)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if len(routes) <= 10:
        ax.legend(fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_cost_hist(samples: np.ndarray, *, alpha: float = 0.95, title: str = "",
                   save: Optional[Path] = None):
    import matplotlib.pyplot as plt
    from .stochastic import cvar
    s = np.asarray(samples)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=30, color="steelblue", alpha=0.8)
    cv = cvar(s, alpha)
    ax.axvline(s.mean(), color="k", ls="--", label=f"E[c+Q]={s.mean():.0f}")
    ax.axvline(cv, color="crimson", ls="-", label=f"CVaR$_{{{alpha}}}$={cv:.0f}")
    ax.set_title(title or "Distribución del costo total con recurso")
    ax.set_xlabel("c + Q"); ax.set_ylabel("frecuencia"); ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_convergence(conv_log, *, gap: float = None, n: int = None, save: Optional[Path] = None):
    """Curva incumbente/cota del Branch & Cut. ``conv_log`` = [(t, best, bound), ...]."""
    import matplotlib.pyplot as plt
    if not conv_log:
        return None
    t = [c[0] for c in conv_log]; best = [c[1] for c in conv_log]; bnd = [c[2] for c in conv_log]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, bnd, color="tab:blue", label="Cota inferior")
    ax.plot(t, best, color="crimson", marker=".", label="Incumbente")
    ax.fill_between(t, bnd, best, color="gold", alpha=0.3, label="Brecha")
    ttl = "Convergencia Branch & Cut"
    if n is not None:
        ttl += f" · n={n}"
    if gap is not None:
        ttl += f" · gap={gap*100:.1f}%"
    ax.set_title(ttl); ax.set_xlabel("tiempo (s)"); ax.set_ylabel("costo determinista")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_training_curve(history: List[float], *, ylabel: str = "pérdida",
                        title: str = "", save: Optional[Path] = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, color="tab:purple")
    ax.set_xlabel("paso/época"); ax.set_ylabel(ylabel)
    ax.set_title(title or "Curva de entrenamiento")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_comparison(df, *, save: Optional[Path] = None):
    """Barras por (tamaño, solver) para las métricas clave."""
    import matplotlib.pyplot as plt
    from .metrics import aggregate_by_size
    agg = aggregate_by_size(df)
    panels = [("expected_cost", "E[c]"), ("expected_total", "E[c+Q]"),
              ("feasibility", "Factibilidad"), ("cvar", "CVaR"),
              ("n_vehicles", "Vehículos")]
    sizes = sorted(agg["size"].unique())
    solvers = sorted(agg["solver"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    width = 0.8 / max(1, len(solvers))
    for ax, (col, lbl) in zip(axes, panels):
        for si, sv in enumerate(solvers):
            d = agg[agg["solver"] == sv].set_index("size").reindex(sizes)
            x = np.arange(len(sizes)) + si * width
            ax.bar(x, d[col].values, width, label=sv)
        ax.set_xticks(np.arange(len(sizes)) + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels(sizes); ax.set_title(lbl); ax.set_xlabel("clientes")
    axes[0].legend(fontsize=7)
    fig.suptitle("Comparación de paradigmas por tamaño")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def plot_tradeoff(df, *, size: int, save: Optional[Path] = None):
    """Tradeoff costo–factibilidad–flota (burbuja = nº de vehículos). La región ideal
    es arriba-izquierda (bajo costo, alta factibilidad)."""
    import matplotlib.pyplot as plt
    sub = df[df["size"] == size].groupby("solver").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in sub.iterrows():
        ax.scatter(r["expected_cost"], r["feasibility"], s=60 + 40 * r["n_vehicles"],
                   alpha=0.6)
        ax.annotate(f"{r['solver']}\n({r['n_vehicles']:.0f} veh)",
                    (r["expected_cost"], r["feasibility"]), fontsize=7,
                    ha="center", va="center")
    ax.set_xlabel("E[c] (menor es mejor →)")
    ax.set_ylabel("Factibilidad (↑ mejor)")
    ax.set_title(f"Tradeoff costo–factibilidad–flota (n={size}); ideal = arriba-izq.")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def save_all_comparison(df, env, slug: str = "cross"):
    """Genera y guarda el set de figuras comparativas estándar."""
    d = _ensure(env, slug)
    plot_comparison(df, save=d / "comparison_metrics.png")
    for s in sorted(df["size"].unique()):
        plot_tradeoff(df, size=int(s), save=d / f"tradeoff_n{int(s)}.png")
    print(f"[viz] figuras comparativas -> {d}")
