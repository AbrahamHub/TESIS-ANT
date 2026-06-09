"""Agregación de métricas entre instancias y tablas de resumen.

Complementa ``stochastic.py`` (que produce la métrica por-instancia) con la
agregación por tamaño y la construcción de tablas/DataFrames que consumen
``run_experiment.py`` y ``viz.py``. Mantiene un único lugar para los nombres de
columna y el orden, de modo que los 5 métodos reporten de forma homogénea.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Columnas canónicas reportadas por todos los métodos (orden fijo).
METRIC_COLUMNS = [
    "solver", "size", "instance",
    "det_cost",        # costo determinista óptimo (objetivo del MIP / longitud nominal)
    "expected_cost",   # E[c] tiempo de viaje estocástico
    "expected_total",  # E[c + Q] con recurso de 2ª etapa
    "cvar",            # CVaR_alpha(c + Q)
    "feasibility",     # tasa de factibilidad [0, 1]
    "cvr",             # tasa de violación (%)
    "robustness",      # std de c entre realizaciones
    "tw_violations",   # promedio de ventanas violadas
    "runtime",         # segundos de cómputo del solver
    "gap",             # brecha de optimalidad MIP (0 = óptimo probado)
    "n_vehicles",      # vehículos usados
]


def row_from_solution(solver: str, size: int, instance: int, sol) -> Dict:
    """Aplana un ``vrp_bench.core.Solution`` a una fila de métricas canónica."""
    ex = sol.extras or {}
    used = sum(1 for r in sol.routes if len(r) > 0)
    return {
        "solver": solver,
        "size": size,
        "instance": instance,
        "det_cost": float(ex.get("det_cost", float("nan"))),
        "expected_cost": float(ex.get("expected_cost", sol.total_cost)),
        "expected_total": float(ex.get("expected_total", sol.total_cost)),
        "cvar": float(ex.get("cvar", float("nan"))),
        "feasibility": float(sol.feasibility),
        "cvr": float(sol.cvr),
        "robustness": float(sol.robustness),
        "tw_violations": float(ex.get("tw_violations", float("nan"))),
        "runtime": float(sol.runtime),
        "gap": float(ex.get("gap", float("nan"))),
        "n_vehicles": int(used),
    }


def to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in METRIC_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[METRIC_COLUMNS]


def aggregate_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios por (solver, size) con desviación del costo esperado."""
    num = df.select_dtypes(include=[np.number]).columns.drop(
        ["instance", "size"], errors="ignore")
    agg = df.groupby(["solver", "size"])[list(num)].mean().reset_index()
    std = (
        df.groupby(["solver", "size"])["expected_cost"].std()
        .reset_index().rename(columns={"expected_cost": "expected_cost_std"})
    )
    return agg.merge(std, on=["solver", "size"], how="left")
