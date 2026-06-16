"""Agregación de métricas y validación estadística.

Define el esquema de columnas canónico que reportan los cinco paradigmas, la
agregación por (solver, tamaño) y las pruebas estadísticas del anteproyecto
(Fase 4): ANOVA de un factor con verificación de supuestos (normalidad de
Shapiro–Wilk, homocedasticidad de Levene) y, cuando no se cumplen —dada la cola
pesada de los costos de ruteo—, alternativas no paramétricas (Friedman para diseño
de bloques por instancia + post-hoc de Wilcoxon pareado con corrección de Holm).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "solver", "paradigm", "size", "instance", "seed",
    "det_cost",        # costo determinista (objetivo del MIP / longitud nominal); NaN si N/A
    "expected_cost",   # E[c]  tiempo de viaje estocástico
    "expected_total",  # E[c+Q]  con recurso de 2ª etapa
    "cvar",            # CVaR_alpha(c+Q)
    "feasibility",     # tasa de factibilidad [0, 1]
    "cvr",             # tasa de violación (%)
    "robustness",      # std de c entre realizaciones
    "tw_violations",   # promedio de ventanas violadas
    "runtime",         # segundos de cómputo del solver (inferencia para NCO)
    "train_time_s",    # costo de entrenamiento amortizado (NCO/EHBG); NaN si N/A
    "gap",             # brecha MIP (0 = óptimo probado); NaN si N/A
    "n_vehicles",      # vehículos/rutas usadas
]


def row_from_solution(solver: str, paradigm: int, size: int, instance: int,
                      seed: int, sol) -> Dict:
    """Aplana un ``vrp_bench.core.Solution`` a una fila canónica."""
    ex = sol.extras or {}
    used = sum(1 for r in sol.routes if len(r) > 0)
    return {
        "solver": solver,
        "paradigm": paradigm,
        "size": size,
        "instance": instance,
        "seed": seed,
        "det_cost": float(ex.get("det_cost", np.nan)),
        "expected_cost": float(ex.get("expected_cost", sol.total_cost)),
        "expected_total": float(ex.get("expected_total", sol.total_cost)),
        "cvar": float(ex.get("cvar", np.nan)),
        "feasibility": float(sol.feasibility),
        "cvr": float(sol.cvr),
        "robustness": float(sol.robustness),
        "tw_violations": float(ex.get("tw_violations", np.nan)),
        "runtime": float(sol.runtime),
        "train_time_s": float(ex.get("train_time_s", np.nan)),
        "gap": float(ex.get("gap", np.nan)),
        "n_vehicles": int(used),
    }


def to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in METRIC_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[METRIC_COLUMNS]


def aggregate_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios por (solver, size) + desviación de E[c] y E[c+Q] entre instancias."""
    num = df.select_dtypes(include=[np.number]).columns.drop(
        ["instance", "size", "seed", "paradigm"], errors="ignore")
    agg = df.groupby(["solver", "size"])[list(num)].mean().reset_index()
    for col in ("expected_cost", "expected_total", "cvar"):
        std = (df.groupby(["solver", "size"])[col].std()
               .reset_index().rename(columns={col: f"{col}_std"}))
        agg = agg.merge(std, on=["solver", "size"], how="left")
    return agg


def leaderboard(df: pd.DataFrame, by: str = "expected_total") -> pd.DataFrame:
    """Tabla resumen por solver (promedio sobre todo el banco), ordenada por ``by``."""
    num = ["expected_cost", "expected_total", "cvar", "feasibility", "cvr",
           "robustness", "runtime", "n_vehicles"]
    out = df.groupby("solver")[num].mean().reset_index().sort_values(by)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validación estadística (Fase 4 del anteproyecto)
# --------------------------------------------------------------------------- #


def _holm(pvals: List[float]) -> List[float]:
    """Corrección de Holm–Bonferroni para comparaciones múltiples."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def compare_solvers(df: pd.DataFrame, *, metric: str = "expected_total",
                    size: Optional[int] = None, alpha: float = 0.05) -> Dict:
    """Compara los solvers sobre ``metric`` con rigor estadístico.

    Diseño de **bloques por instancia**: cada instancia (mismo seed) es un bloque
    medido por todos los solvers (gracias al CRN). Devuelve un dict con:
      * supuestos: normalidad (Shapiro por solver) y homocedasticidad (Levene);
      * prueba ómnibus paramétrica (ANOVA) y no paramétrica (Friedman);
      * recomendación (cuál usar) según los supuestos;
      * post-hoc Wilcoxon pareado (Holm) entre todos los pares de solvers.
    """
    from scipy import stats

    sub = df if size is None else df[df["size"] == size]
    # Matriz instancias × solvers alineada por (size, instance) — exige diseño completo.
    pivot = sub.pivot_table(index=["size", "instance"], columns="solver",
                            values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")   # bloques medidos por todos los solvers
    solvers = list(pivot.columns)
    groups = [pivot[s].to_numpy() for s in solvers]

    out: Dict = {"metric": metric, "size": size, "solvers": solvers,
                 "n_blocks": int(pivot.shape[0]), "alpha": alpha}
    if pivot.shape[0] < 3 or len(solvers) < 2:
        out["note"] = "insuficientes bloques/solvers para pruebas formales"
        return out

    # Supuestos
    normal = {}
    for s, g in zip(solvers, groups):
        if np.ptp(g) == 0:
            normal[s] = 1.0   # constante (p. ej. feasibilidad 0): trátese como degenerado
        else:
            normal[s] = float(stats.shapiro(g).pvalue)
    levene_p = float(stats.levene(*groups).pvalue) if len(groups) > 1 else 1.0
    all_normal = all(p > alpha for p in normal.values())
    homoscedastic = levene_p > alpha
    out["assumptions"] = {"shapiro_p": normal, "levene_p": levene_p,
                          "all_normal": all_normal, "homoscedastic": homoscedastic}

    # Ómnibus
    try:
        f, p_anova = stats.f_oneway(*groups)
        out["anova"] = {"F": float(f), "p": float(p_anova)}
    except Exception as e:
        out["anova"] = {"error": str(e)}
    try:
        chi, p_fr = stats.friedmanchisquare(*groups)
        out["friedman"] = {"chi2": float(chi), "p": float(p_fr)}
    except Exception as e:
        out["friedman"] = {"error": str(e)}

    out["recommended_test"] = ("ANOVA" if (all_normal and homoscedastic) else "Friedman")

    # Post-hoc Wilcoxon pareado (todos los pares) + Holm
    pairs, praw = [], []
    for a, b in itertools.combinations(solvers, 2):
        x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
        if np.allclose(x, y):
            p = 1.0
        else:
            try:
                p = float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)
            except Exception:
                p = float("nan")
        pairs.append((a, b)); praw.append(p)
    padj = _holm([0.0 if np.isnan(p) else p for p in praw])
    out["posthoc_wilcoxon"] = [
        {"a": a, "b": b, "median_diff": float(np.median(pivot[a] - pivot[b])),
         "p_raw": praw[i], "p_holm": padj[i], "significant": padj[i] < alpha}
        for i, (a, b) in enumerate(pairs)
    ]
    return out


def summarize_comparison(cmp: Dict) -> str:
    """Texto legible del resultado de ``compare_solvers`` (para imprimir en notebook)."""
    if "anova" not in cmp:
        return f"[stats] {cmp.get('note', 'sin resultado')}"
    a = cmp["assumptions"]
    lines = [
        f"Métrica: {cmp['metric']}  |  tamaño: {cmp['size']}  |  bloques: {cmp['n_blocks']}",
        f"Supuestos: normalidad={'sí' if a['all_normal'] else 'no'} "
        f"(Levene p={a['levene_p']:.3g}, homocedástico={'sí' if a['homoscedastic'] else 'no'})",
        f"ANOVA: F={cmp['anova'].get('F', float('nan')):.3f} p={cmp['anova'].get('p', float('nan')):.3g}  |  "
        f"Friedman: χ²={cmp['friedman'].get('chi2', float('nan')):.3f} p={cmp['friedman'].get('p', float('nan')):.3g}",
        f"Prueba recomendada: {cmp['recommended_test']} (α={cmp['alpha']})",
        "Post-hoc Wilcoxon (Holm):",
    ]
    for r in cmp["posthoc_wilcoxon"]:
        mark = "*" if r["significant"] else " "
        lines.append(f"  {mark} {r['a']} vs {r['b']}: Δmed={r['median_diff']:+.1f}  p_holm={r['p_holm']:.3g}")
    return "\n".join(lines)
