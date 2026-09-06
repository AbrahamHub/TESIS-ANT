"""Reporte compacto del estado de las corridas — para pegar donde haga falta.

Uso en Colab (una sola celda, despues del setup de cualquier notebook):

    !python {_path}/scripts/report_runs.py --root "{env.paths.root}"

o directamente desde un notebook que ya tenga `env`:

    from svrplab import reporting            # atajo equivalente
    print(reporting.report(env))

Responde, sin re-resolver nada, las preguntas que deciden si los resultados
sirven: quien corrio que, si alguna instancia fallo, si la busqueda de cada
solver fue disjunta de la evaluacion, si todos usaron el mismo banco y el mismo
protocolo, y que dicen los numeros sobre atribucion y sobre el CVaR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "-"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def report(root: Path) -> str:
    import pandas as pd
    import numpy as np

    root = Path(root)
    res = root / "results"
    out: list[str] = []
    add = out.append

    add("=" * 78)
    add(f"REPORTE DE CORRIDAS  —  raiz: {root}")
    add("=" * 78)

    if not res.is_dir():
        add(f"\nNo existe {res}. ¿Es la raiz correcta? En el notebook: env.paths.root")
        return "\n".join(out)

    # ---------- 1. Que se ejecuto ------------------------------------------
    metas = {}
    for p in sorted(res.rglob("*_run.json")):
        try:
            m = json.loads(p.read_text())
            metas[m.get("solver", p.stem)] = m
        except Exception as e:
            add(f"  (no pude leer {p.name}: {e})")

    add(f"\n--- 1. Solvers con corrida registrada: {len(metas)} ---")
    if not metas:
        add("  NINGUNO. No hay *_run.json bajo results/: las libretas no llegaron")
        add("  a persistir, o la raiz de artefactos no es esta.")
        return "\n".join(out)

    add(f"{'solver':16}{'tamaños':22}{'completa':>9}{'sin fallos':>11}{'fallos':>7}")
    for sv, m in sorted(metas.items()):
        add(f"{sv:16}{str(m.get('sizes', [])):22}"
            f"{str(m.get('complete')):>9}{str(m.get('all_valid')):>11}"
            f"{m.get('n_failures', 0):>7}")

    # ---------- 2. Anti-fuga de escenarios ---------------------------------
    add("\n--- 2. Busqueda disjunta de la evaluacion (anti-fuga) ---")
    fugados = []
    for sv, m in sorted(metas.items()):
        rg = m.get("scenario_ranges") or {}
        ev, se, dj = rg.get("eval"), rg.get("search"), rg.get("disjoint")
        if not rg:
            add(f"  {sv:16} sin registro de rangos (corrida ANTERIOR a la correccion)")
            fugados.append(sv)
            continue
        malo = (not dj) or (ev and se and se[0] < ev[1])
        add(f"  {sv:16} eval={ev}  busqueda={se}  disjunta={dj}"
            f"{'   <-- FUGA' if malo else ''}")
        if malo:
            fugados.append(sv)
    add("  VEREDICTO: " + ("todos limpios" if not fugados else
                           f"REVISAR {', '.join(fugados)}"))

    # ---------- 3. Mismo banco y mismo protocolo ---------------------------
    add("\n--- 3. Piso parejo (mismo banco y mismo protocolo) ---")
    fps, protos = {}, {}
    for sv, m in metas.items():
        for s, h in (m.get("size_fingerprints") or {}).items():
            fps.setdefault(int(s), {})[sv] = h
        pr = m.get("protocol") or {}
        protos[sv] = tuple(pr.get(k) for k in
                           ("realizations", "alpha", "late_penalty",
                            "accident_scale", "vehicle_fixed_cost", "base_seed"))
    discrepantes = [s for s, d in fps.items() if len(set(d.values())) > 1]
    add(f"  banco identico por tamaño : "
        f"{'SI' if not discrepantes else 'NO -> tamaños ' + str(discrepantes)}")
    add(f"  protocolo identico        : "
        f"{'SI' if len(set(protos.values())) <= 1 else 'NO'}")
    if len(set(protos.values())) > 1:
        for sv, t in sorted(protos.items()):
            add(f"     {sv:16} (R,alpha,late,acc,flota,seed) = {t}")

    # ---------- 4. Cobertura y numeros -------------------------------------
    frames = []
    for csv in sorted(res.rglob("*_metrics.csv")):
        try:
            frames.append(pd.read_csv(csv))
        except Exception:
            pass
    if not frames:
        add("\n--- 4. Sin *_metrics.csv legibles ---")
        return "\n".join(out)
    df = pd.concat(frames, ignore_index=True)

    add("\n--- 4. Cobertura: instancias por (solver, tamaño) ---")
    cov = df.pivot_table(index="solver", columns="size", values="instance",
                         aggfunc="count").fillna(0).astype(int)
    add(cov.to_string())

    if "error" in df.columns:
        bad = df["error"].fillna("").astype(str).str.len() > 0
        add(f"\n  instancias con error: {int(bad.sum())}")
        for _, r in df.loc[bad].head(8).iterrows():
            add(f"    {r['solver']} n={r['size']} i={r['instance']}: "
                f"{str(r['error'])[:90]}")
        df = df.loc[~bad]

    add("\n--- 5. Resultados por tamaño (media sobre instancias) ---")
    cols = [c for c in ("expected_cost", "expected_total", "cvar", "feasibility",
                        "cvr", "n_vehicles", "runtime", "search_budget")
            if c in df.columns]
    agg = df.groupby(["size", "solver"])[cols].mean().round(2)
    add(agg.to_string())

    # ---------- 6. Atribucion: la red frente al mismo presupuesto ----------
    add("\n--- 6. Atribucion: ehbg-facs frente a facs-dist (mismo presupuesto) ---")
    if {"ehbg-facs", "facs-dist"} <= set(df.solver.unique()):
        for s in sorted(df["size"].unique()):
            d = df[df["size"] == s]
            a = d[d.solver == "ehbg-facs"]["expected_total"].mean()
            b = d[d.solver == "facs-dist"]["expected_total"].mean()
            ba = d[d.solver == "ehbg-facs"]["search_budget"].mean() \
                if "search_budget" in d else float("nan")
            bb = d[d.solver == "facs-dist"]["search_budget"].mean() \
                if "search_budget" in d else float("nan")
            gan = 100 * (b - a) / b if b else float("nan")
            add(f"  n={int(s):<4} ehbg-facs={_fmt(a)}  facs-dist={_fmt(b)}  "
                f"ventaja de la red={_fmt(gan,2)} %  "
                f"presupuestos={_fmt(ba,0)}/{_fmt(bb,0)}"
                f"{'  (DESIGUALES)' if ba == ba and bb == bb and ba != bb else ''}")
        add("  Si la ventaja no es positiva, la propuesta no supera al prior de")
        add("  distancia con el mismo presupuesto: hay que reportarlo como tal.")
    else:
        add("  falta ehbg-facs o facs-dist: sin este control ningun resultado")
        add("  positivo es atribuible a la GFlowNet.")

    # ---------- 7. ¿El CVaR aporta algo? -----------------------------------
    add("\n--- 7. ¿El CVaR aporta informacion sobre la media? ---")
    if {"cvar", "expected_total"} <= set(df.columns):
        d = df[df["expected_total"] > 0]
        brecha = (100 * (d["cvar"] - d["expected_total"]) / d["expected_total"])
        add(f"  brecha CVaR - E[c+Q]: media {_fmt(brecha.mean(),4)} %  "
            f"max {_fmt(brecha.max(),4)} %")
        add("  " + ("INDISTINGUIBLE de la media: las afirmaciones sobre riesgo de"
                    " cola no son contrastables en este regimen."
                    if brecha.mean() < 1 else
                    "El CVaR discrimina en este regimen."))

    add("\n" + "=" * 78)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="raiz de artefactos (en el notebook: env.paths.root)")
    ap.add_argument("--save", default=None, help="ademas, guarda el texto aqui")
    a = ap.parse_args()
    txt = report(Path(a.root))
    print(txt)
    if a.save:
        Path(a.save).write_text(txt)
        print(f"\nguardado en {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
