"""Análisis de sensibilidad y auditorías de validez del protocolo.

Dos preguntas del dictamen de revisión se responden aquí, y ninguna de las dos
exige volver a resolver las instancias: bastan las rutas ya persistidas.

1. **¿Discrimina el CVaR?** A la tasa oficial de accidentes de SVRPBench
   (λ ≈ 1.6·10⁻⁴) el CVaR coincide con la media dentro del 0.02 %: la recompensa
   sensible al riesgo optimiza algo numéricamente indistinguible del costo medio,
   y las hipótesis sobre robustez de cola no son contrastables. ``risk_regime_sweep``
   re-puntúa las MISMAS rutas a varias escalas de accidentes y localiza el régimen
   donde el CVaR sí separa. Reportar ambos regímenes —fidelidad estricta (×1) y
   estrés declarado (×k)— es lo que vuelve honesta la afirmación.

2. **¿Cuánto infla el resultado buscar sobre los escenarios de evaluación?**
   ``leakage_bias`` corre el mismo solver con búsqueda disjunta y con búsqueda
   dentro de la muestra de evaluación, y mide la diferencia. Es la cuantificación
   del sesgo de selección: sirve para declarar en la tesis cuánto valía la fuga
   que se corrigió, en vez de solo afirmar que se corrigió.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import stochastic
from .protocol import Protocol


# --------------------------------------------------------------------------- #
# 1) Régimen de riesgo: ¿a qué escala de accidentes el CVaR deja de ser la media?
# --------------------------------------------------------------------------- #


def risk_regime_sweep(instances, routes_by_key: Dict[str, list], proto: Protocol,
                      scales: Iterable[float] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
                      *, solver_name: str = "", verbose: bool = True) -> pd.DataFrame:
    """Re-puntúa rutas ya calculadas a varias ``accident_scale``.

    Parámetros
    ----------
    instances : ``{"size:idx": Instance}`` o lista de ``(key, Instance)``.
    routes_by_key : rutas persistidas (``runner.load_routes``), misma clave.
    proto : protocolo homologado (aporta R, α, penalización y cubetas).
    scales : multiplicadores de la tasa de accidentes. ``1.0`` = fiel a SVRPBench.

    Devuelve una fila por (escala, instancia) con ``expected_total``, ``cvar`` y
    ``cvar_gap_pct`` = 100·(CVaR − E[c+Q]) / E[c+Q]. **La lectura clave es
    ``cvar_gap_pct``**: mientras esté por debajo de ~1 %, el CVaR no aporta
    información más allá de la media y ninguna conclusión sobre riesgo de cola
    puede sostenerse en ese régimen.
    """
    if isinstance(instances, dict):
        items = list(instances.items())
    else:
        items = list(instances)

    rows: List[dict] = []
    for scale in scales:
        for key, inst in items:
            routes = routes_by_key.get(key)
            if not routes:
                continue
            sc = stochastic.score_routes(
                inst, routes, seed=int(inst.metadata.get("seed", 0)),
                num_realizations=proto.realizations, alpha=proto.alpha,
                late_penalty=proto.late_penalty, accident_scale=float(scale),
                n_buckets=proto.n_buckets, r_offset=0)
            size, idx = key.split(":")
            gap = (100.0 * (sc.cvar - sc.expected_total) / sc.expected_total
                   if sc.expected_total else np.nan)
            rows.append({
                "solver": solver_name, "accident_scale": float(scale),
                "size": int(size), "instance": int(idx),
                "expected_cost": sc.expected_cost, "expected_total": sc.expected_total,
                "cvar": sc.cvar, "cvar_gap_pct": gap,
                "total_std": float(np.std(sc.total_samples)),
                "feasibility": sc.feasibility, "cvr": sc.cvr,
            })
    df = pd.DataFrame(rows)
    if verbose and len(df):
        _print_regime_verdict(df)
    return df


def _print_regime_verdict(df: pd.DataFrame, threshold: float = 1.0) -> None:
    """Dice explícitamente en qué escalas el CVaR es informativo."""
    agg = df.groupby("accident_scale")["cvar_gap_pct"].mean()
    print("Separación CVaR vs media por escala de accidentes:")
    for scale, gap in agg.items():
        verdict = "informativo" if gap >= threshold else "INDISTINGUIBLE de la media"
        print(f"  ×{scale:<6g}  brecha CVaR−E[c+Q] = {gap:6.3f} %   {verdict}")
    good = agg[agg >= threshold]
    if good.empty:
        print(f"\n  VEREDICTO: en ninguna escala probada el CVaR separa más de "
              f"{threshold} %. Las afirmaciones sobre robustez de cola NO son "
              f"contrastables con este barrido; amplía las escalas.")
    else:
        print(f"\n  VEREDICTO: el CVaR discrimina desde ×{good.index.min():g}. "
              f"Reporta ×1 (fidelidad a SVRPBench) y ×{good.index.min():g} "
              f"(estrés declarado) por separado; las hipótesis de riesgo solo "
              f"pueden contrastarse en el segundo régimen.")


# --------------------------------------------------------------------------- #
# 2) Sesgo de selección: cuánto valía buscar sobre los escenarios de evaluación
# --------------------------------------------------------------------------- #


def leakage_bias(solver_factory, instances, proto: Protocol, *,
                 solver_name: str = "solver", verbose: bool = True) -> pd.DataFrame:
    """Cuantifica el sesgo de seleccionar sobre la muestra de evaluación.

    ``solver_factory(search_r_offset)`` debe devolver un solver nuevo configurado con
    ese desplazamiento de escenarios de búsqueda. Se corren dos configuraciones sobre
    las MISMAS instancias:

      * **disjunta** (correcta): ``search_r_offset = proto.realizations``.
      * **fugada** (sesgada): ``search_r_offset = 0`` — la búsqueda ve exactamente
        los escenarios con los que después se mide.

    La diferencia en ``expected_total``/``cvar`` es la inflación atribuible al sesgo
    de selección, no a la calidad del método. Un valor grande justifica en la tesis
    por qué la corrección era necesaria; uno pequeño acota la severidad del defecto.
    """
    rows: List[dict] = []
    items = list(instances.items()) if isinstance(instances, dict) else list(instances)

    for label, offset in (("disjunta", int(proto.realizations)), ("fugada", 0)):
        solver = solver_factory(offset)
        for key, inst in items:
            sol = solver.solve(inst, num_realizations=proto.realizations)
            sc = stochastic.score_routes(
                inst, sol.routes, seed=int(inst.metadata.get("seed", 0)),
                **proto.eval_kwargs())
            size, idx = key.split(":")
            rows.append({"solver": solver_name, "busqueda": label,
                         "search_r_offset": offset, "size": int(size),
                         "instance": int(idx), "expected_total": sc.expected_total,
                         "cvar": sc.cvar, "feasibility": sc.feasibility})
    df = pd.DataFrame(rows)
    if verbose and len(df):
        _print_leakage_verdict(df)
    return df


def _print_leakage_verdict(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index=["size", "instance"], columns="busqueda",
                         values="expected_total")
    if not {"disjunta", "fugada"} <= set(piv.columns):
        return
    infl = 100.0 * (piv["disjunta"] - piv["fugada"]) / piv["disjunta"]
    print(f"\nSesgo de selección sobre E[c+Q] ({len(piv)} instancias):")
    print(f"  búsqueda disjunta (correcta): {piv['disjunta'].mean():9.1f}")
    print(f"  búsqueda fugada   (sesgada):  {piv['fugada'].mean():9.1f}")
    print(f"  inflación media del resultado fugado: {infl.mean():.2f} % "
          f"(mediana {infl.median():.2f} %)")
    if infl.mean() > 1.0:
        print("  LECTURA: la fuga inflaba el resultado de forma material. "
              "Cualquier comparación previa a la corrección es inválida.")
    else:
        print("  LECTURA: la inflación es pequeña en este banco, pero la corrección "
              "sigue siendo necesaria: el sesgo crece con el presupuesto de búsqueda "
              "y no es acotable a priori.")


# --------------------------------------------------------------------------- #
# 3) Auditoría del piso parejo entre solvers
# --------------------------------------------------------------------------- #


def audit_runs(run_meta: Dict[str, dict], *, strict: bool = False) -> pd.DataFrame:
    """Audita los ``*_run.json`` de todos los solvers antes de compararlos.

    Verifica cuatro cosas que invalidan una comparación si fallan:
      1. **Mismo banco** por tamaño (``size_fingerprints``).
      2. **Mismo protocolo de evaluación** (R, α, penalización, escala, flota).
      3. **Búsqueda disjunta** de la evaluación en todo solver que busque.
      4. **Corrida completa y sin fallos** (``all_valid``).

    Con ``strict=True`` lanza ``AssertionError`` ante cualquier incumplimiento; en
    modo normal devuelve la tabla y marca los problemas para que el autor decida.
    """
    rows = []
    for name, m in sorted(run_meta.items()):
        proto = m.get("protocol", {})
        sr = m.get("scenario_ranges", {})
        rows.append({
            "solver": name,
            "R_eval": proto.get("realizations"),
            "alpha": proto.get("alpha"),
            "late_penalty": proto.get("late_penalty"),
            "accident_scale": proto.get("accident_scale"),
            "vehicle_fixed_cost": proto.get("vehicle_fixed_cost"),
            "busqueda_disjunta": sr.get("disjoint"),
            "rango_eval": str(sr.get("eval")),
            "rango_busqueda": str(sr.get("search")),
            "completa": m.get("complete"),
            "sin_fallos": m.get("all_valid"),
            "n_fallos": m.get("n_failures", 0),
            "banco": m.get("bank_fingerprint"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[audit] no hay corridas que auditar todavía")
        return df

    problems: List[str] = []
    for col, label in [("R_eval", "realizaciones de evaluación"),
                       ("alpha", "nivel de CVaR"),
                       ("late_penalty", "penalización de recurso"),
                       ("accident_scale", "escala de accidentes"),
                       ("vehicle_fixed_cost", "costo de flota")]:
        vals = set(df[col].dropna().unique())
        if len(vals) > 1:
            problems.append(f"{label} difiere entre solvers: {sorted(vals)}")

    leaky = df.loc[df["busqueda_disjunta"] == False, "solver"].tolist()  # noqa: E712
    if leaky:
        problems.append("búsqueda NO disjunta de la evaluación en: " + ", ".join(leaky))

    failed = df.loc[df["sin_fallos"] == False, "solver"].tolist()  # noqa: E712
    if failed:
        problems.append("corridas incompletas o con fallos: " + ", ".join(failed))

    # El banco puede diferir legítimamente si un solver corrió menos tamaños
    # (p. ej. los exactos limitados por licencia); por eso se compara por tamaño
    # en el notebook con `size_fingerprints`, no la huella global.
    print("=== Auditoría de piso parejo ===")
    if problems:
        print("PROBLEMAS DETECTADOS (la comparación no es válida hasta resolverlos):")
        for p in problems:
            print(f"  - {p}")
        if strict:
            raise AssertionError("; ".join(problems))
    else:
        print("Todo en orden: mismo protocolo de evaluación, búsqueda disjunta en "
              "todos los solvers, y ninguna corrida con fallos.")
    return df
