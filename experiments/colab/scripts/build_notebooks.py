"""Genera los notebooks .ipynb del pipeline (uno por paradigma + setup + comparación).

Construye el JSON nbformat v4 a mano (sin dependencias). Ejecutar:

    python experiments/colab/scripts/build_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


# --------------------------------------------------------------------------- #
# Bloques reutilizables
# --------------------------------------------------------------------------- #

SETUP_CODE = '''
# === Configuración del entorno (ejecuta esta celda primero) =================
# Requiere: (a) el paquete `svrplab` (carpeta experiments/colab del repo de tesis)
#           (b) el repo oficial de SVRPBench (se clona solo en bootstrap.init()).
REPO_URL  = "https://github.com/AbrahaHub/TESIS-ANT"   # <-- EDITA si tu repo difiere
USE_DRIVE = True   # persistir banco/resultados/modelos en Google Drive (recomendado)

import os, sys, subprocess

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive no disponible (¿ejecutas local?):", e)

def _find_svrplab():
    cands = ["/content/drive/MyDrive/TESIS-ANT/experiments/colab",
             "/content/TESIS-ANT/experiments/colab",
             os.path.join(os.getcwd(), "experiments", "colab"),
             os.getcwd()]
    for c in cands:
        if os.path.isdir(os.path.join(c, "svrplab")):
            return c
    return None

_path = _find_svrplab()
if _path is None:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/TESIS-ANT"], check=False)
    _path = "/content/TESIS-ANT/experiments/colab"
sys.path.insert(0, _path)
print("svrplab en:", _path)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy", "pandas",
                "matplotlib", "scikit-learn", "pillow", "tqdm"], check=False)
# torch ya viene en Colab. gurobipy solo se instala en el notebook del paradigma 1.

from svrplab import bootstrap, protocol, data, runner, metrics, viz
env   = bootstrap.init()        # GPU + repo oficial SVRPBench + rutas (Drive si está montado)
proto = protocol.DEFAULT
print("device:", env.device, "| raíz de artefactos:", env.paths.root)
'''.strip("\n")

CONFIG_CODE = '''
# === Configuración del experimento (IDÉNTICA en los 5 notebooks) ============
# Para garantizar el "piso parejo", TODOS los notebooks deben usar los MISMOS
# SIZES y N_INSTANCES: así resuelven exactamente el mismo banco de instancias.
# (Las semillas por instancia dependen solo de (base_seed, tamaño, índice), así
#  que el banco con N_INSTANCES=5 es un PREFIJO exacto del banco con 30.)
SIZES       = [10, 20, 50, 100, 200, 300]           # clientes. Extiende a [50,100,200,300] (ver notas).
N_INSTANCES = 5   # 30 (rigor estadístico). Corrida rápida: pon 5.

# Paralelismo de las fases CPU (Gurobi, ACO/Tabu, evaluador CRN): procesos fork —
# el GIL impide escalar con hilos. vCPU típicas en Colab: T4≈2, L4≈8, A100≈12.
import os
N_JOBS = max(1, os.cpu_count() or 1)
print(f"N_JOBS = {N_JOBS} procesos paralelos (vCPU detectadas)")

bank = data.load_bank(env.paths.instances, SIZES, N_INSTANCES,
                      base_seed=proto.base_seed, capacity_mode=proto.capacity_mode, verbose=True)
print({s: len(v) for s, v in bank.items()}, "instancias por tamaño")
if N_INSTANCES < 10:
    print("ADVERTENCIA: con N_INSTANCES=5 el Wilcoxon pareado NO puede alcanzar p<0.05 "
          "(mínimo teórico bilateral con n=5: 0.0625). Es una corrida exploratoria; "
          "para las conclusiones de la tesis usa N_INSTANCES=30.")
'''.strip("\n")


def header(title, subtitle, body):
    return md(f"# {title}\n\n**{subtitle}**\n\n{body}")


# --------------------------------------------------------------------------- #
# 00 — Setup y banco canónico
# --------------------------------------------------------------------------- #

nb00 = notebook([
    header("Pipeline EHBG-FACS · 00 · Setup y banco de instancias",
           "Prepara el entorno y construye el banco canónico que los 5 paradigmas comparten.",
           "Este pipeline compara, bajo condiciones **homologadas**, los cinco paradigmas "
           "del anteproyecto sobre **SVRPBench**: (1) Exactos Branch & Cut, (2) Metaheurísticas "
           "(ACO/Tabu), (3) NCO supervisado (Attention Model), (4) NCO por RL (POMO+AM) y "
           "(5) la propuesta **EHBG-FACS**. Todos resuelven el **mismo** banco de instancias y "
           "se puntúan con el **mismo** evaluador estocástico (CRN + recurso de 2ª etapa + CVaR), "
           "lo que garantiza una comparación replicable y justa."),
    md("## 1. Entorno\nMonta Drive (persistencia), localiza `svrplab`, clona el repo oficial de "
       "SVRPBench y detecta la GPU."),
    code(SETUP_CODE),
    md("## 2. Protocolo homologado\nFuente única de verdad de las condiciones experimentales "
       "(idénticas para los 5 paradigmas)."),
    code('import pprint; pprint.pprint(proto.as_dict())'),
    md("## 3. Banco canónico de instancias\nSe genera **una sola vez** con semillas fijas y se "
       "cachea en `data/instances/`. Los notebooks de paradigma lo cargan tal cual. Reutiliza las "
       "primitivas oficiales de SVRPBench (`city.City`, `time_windows_generator`)."),
    code(CONFIG_CODE),
    md("## 4. Inspección visual de una instancia\nDepósito (rojo) en el centroide; clientes "
       "coloreados por la apertura de su ventana de tiempo. La figura queda exportada en "
       "`figures/00_setup/` (Drive)."),
    code('inst = bank[SIZES[0]][0]\n'
         'fig = viz.plot_instance(inst, title=f"Instancia TWCVRP n={SIZES[0]} (seed={inst.metadata[\'seed\']})")\n'
         'viz.save_show(fig, env, "00_setup", f"instancia_n{SIZES[0]}")'),
    md("## 5. Vectores estocásticos de SVRPBench\nVisualiza los cuatro mecanismos de "
       "incertidumbre (congestión por mezcla gaussiana, retraso log-normal, accidentes de Poisson, "
       "ventanas residencial/comercial)."),
    code('import numpy as np, matplotlib.pyplot as plt\n'
         'from svrplab import stochastic as S\n'
         't = np.linspace(0, 1440, 600)\n'
         'fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))\n'
         'ax[0].plot(t, [S.time_factor(x) for x in t]); ax[0].set_title("Congestión time_factor(t)")\n'
         'ax[0].axvline(480, ls="--", c="k"); ax[0].axvline(1020, ls="--", c="k")\n'
         'lam = 0.05*np.array([S.normal_distribution(x,1260,120) for x in t])\n'
         'ax[1].plot(t, lam, c="purple"); ax[1].set_title("Tasa Poisson accidentes λ(t)")\n'
         'tw = np.asarray(inst.time_windows)[1:]\n'
         'ax[2].hist(tw[:,0], bins=20, color="teal"); ax[2].set_title("Apertura de ventanas")\n'
         'for a in ax: a.set_xlabel("min del día")\n'
         'plt.tight_layout()\n'
         'viz.save_show(fig, env, "00_setup", "vectores_estocasticos")'),
    md("---\n**Listo.** Ejecuta ahora los notebooks `01`…`05` (en cualquier orden) y, al final, "
       "`06_comparacion_y_estadistica` para la tabla comparativa y las pruebas ANOVA/Wilcoxon. "
       "Mantén `USE_DRIVE=True` y los mismos `SIZES`/`N_INSTANCES` en todos."),
])


# --------------------------------------------------------------------------- #
# 01 — Exactos
# --------------------------------------------------------------------------- #

nb01 = notebook([
    header("Pipeline EHBG-FACS · 01 · Métodos Exactos (Branch & Cut)",
           "Paradigma 1 — cotas exactas con Gurobi: costo (CVRP) y cumplimiento (CVRPTW soft).",
           "Dos baselines exactos complementarios: **`exact-bc`** resuelve el CVRP sobre el "
           "**tiempo de viaje nominal** (no dirigido, cortes RCI/DFJ como *lazy constraints*) — "
           "la referencia de costo que ignora ventanas; **`exact-bc-tw`** resuelve el CVRPTW "
           "dirigido (MTZ) con **ventanas soft** — la referencia que *sí busca satisfacer las "
           "restricciones* en el plan nominal. Ambos arrancan de una incumbente golosa (MIP "
           "start) y siempre devuelven ruta; su intratabilidad aparece al crecer *n*."),
    code(SETUP_CODE),
    md("## Gurobi\nInstala `gurobipy` (licencia restringida incluida; la celda siguiente detecta "
       "tu licencia académica si la subiste a Drive)."),
    code('import subprocess, sys\n'
         'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gurobipy"], check=False)\n'
         'import gurobipy; print("Gurobi", gurobipy.gurobi.version())'),
    code(CONFIG_CODE),
    md("## Licencia de Gurobi y alcance del exacto\nLa licencia restringida de `gurobipy` "
       "(≤2000 vars) cubre: `exact-bc` (no dirigido, ~n²/2 vars) hasta n≈62 y `exact-bc-tw` "
       "(dirigido, ~n² vars) hasta n≈42. Con **licencia académica WLS** (gratuita: "
       "portal.gurobi.com → *WLS license*), guarda tu `gurobi.lic` en Drive y esta celda la "
       "detecta y amplía el alcance automáticamente. El exacto sigue siendo la cota de escala "
       "pequeña del Cuadro 1 (*Muy Baja, <100 nodos*): a n grande el límite práctico es el "
       "tiempo, no solo la licencia."),
    code('GUROBI_LIC = "/content/drive/MyDrive/gurobi.lic"   # <-- súbela a Drive si la tienes\n'
         'if os.path.exists(GUROBI_LIC):\n'
         '    os.environ["GRB_LICENSE_FILE"] = GUROBI_LIC\n'
         '    EXACT_MAX_N, TW_MAX_N = 100, 50   # con más tiempo/cómputo puedes subirlos\n'
         '    print("Licencia académica detectada:", GUROBI_LIC)\n'
         'else:\n'
         '    EXACT_MAX_N, TW_MAX_N = 50, 20    # límites de la licencia restringida pip\n'
         '    print("Licencia restringida pip: exact-bc hasta n=50, exact-bc-tw hasta n=20.")\n'
         'bank_exact = {s: v for s, v in bank.items() if s <= EXACT_MAX_N}\n'
         'bank_tw    = {s: v for s, v in bank.items() if s <= TW_MAX_N}\n'
         'print("exact-bc:", sorted(bank_exact), "| exact-bc-tw:", sorted(bank_tw),\n'
         '      "| omitidos:", [s for s in SIZES if s > EXACT_MAX_N])'),
    md("## Resolver `exact-bc` (CVRP, referencia de costo)\nValida cada solución post-resolución "
       "(capacidad, cobertura, gap≥0) y arranca el B&C desde una **incumbente golosa NN+capacidad** "
       "(MIP start): incumbentes tempranos y **siempre** devuelve ruta aunque el límite de tiempo "
       "expire (si no hubiera incumbente, cae a la golosa marcada `fallback=True`). "
       "`n_jobs=N_JOBS` resuelve instancias en procesos paralelos, cada uno con Gurobi "
       "`Threads=1` (obligatorio para la corrección de los cortes perezosos). Si tu licencia "
       "WLS limita sesiones concurrentes, usa `n_jobs=1`."),
    code('from svrplab.solvers.exact_bc import ExactBranchCut, ExactBranchCutTW\n'
         'import pandas as pd\n'
         'solver = ExactBranchCut(time_limit=120.0, verbose=False)\n'
         'df_bc = runner.run_solver(solver, "exact-bc", bank_exact, env, proto, verbose=True,\n'
         '                          n_jobs=N_JOBS)\n'
         'df_bc'),
    md("## Resolver `exact-bc-tw` (CVRPTW soft — **busca satisfacer las ventanas**)\n"
       "Formulación dirigida MTZ que penaliza la tardanza nominal `L_j` en el objetivo "
       "(ventanas *soft*: por el caveat de escala de SVRPBench las ventanas duras son casi "
       "insatisfacibles desde n≈20). Aísla el efecto de la estocasticidad: la infactibilidad "
       "residual bajo ξ es atribuible a la incertidumbre, no a ignorar las restricciones."),
    code('solver_tw = ExactBranchCutTW(time_limit=120.0, tw_penalty=proto.late_penalty,\n'
         '                             verbose=False)\n'
         'df_tw = runner.run_solver(solver_tw, "exact-bc-tw", bank_tw, env, proto, verbose=True,\n'
         '                          n_jobs=N_JOBS)\n'
         'df = pd.concat([df_bc, df_tw], ignore_index=True)\n'
         'df_tw'),
    md("## Métricas agregadas y figuras\nTodas las figuras quedan en `figures/01_exact/`."),
    code('agg = metrics.aggregate_by_size(df); display(agg)\n'
         '# Ruta + convergencia B&C de la primera instancia del menor tamaño\n'
         'inst = bank_exact[min(bank_exact)][0]\n'
         'sol = solver.solve(inst, num_realizations=proto.realizations)\n'
         'fig = viz.plot_routes(inst, sol.routes, title=f"exact-bc · n={min(bank_exact)}")\n'
         'viz.save_show(fig, env, "01_exact", f"rutas_exact_bc_n{min(bank_exact)}")\n'
         'fig = viz.plot_convergence(sol.extras.get("convergence_log", []), gap=sol.extras.get("gap"),\n'
         '                           n=min(bank_exact))\n'
         'viz.save_show(fig, env, "01_exact", f"convergencia_bc_n{min(bank_exact)}")\n'
         'sol_tw = solver_tw.solve(bank_tw[min(bank_tw)][0], num_realizations=proto.realizations)\n'
         'fig = viz.plot_routes(bank_tw[min(bank_tw)][0], sol_tw.routes,\n'
         '                      title=f"exact-bc-tw · n={min(bank_tw)}")\n'
         'viz.save_show(fig, env, "01_exact", f"rutas_exact_bc_tw_n{min(bank_tw)}")\n'
         'fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "01_exact", "metricas_por_tamano")'),
    md("**Interpretación.** `exact-bc` minimiza el costo nominal e ignora las ventanas: menor "
       "`E[c]`, factibilidad baja — referencia de costo. `exact-bc-tw` sí busca satisfacer las "
       "ventanas (tardanza nominal penalizada): menor `E[c+Q]`/tardanza a cambio de más costo "
       "de viaje — referencia de cumplimiento determinista. La brecha entre ambos aísla el "
       "precio de las restricciones; la brecha de `exact-bc-tw` frente a la factibilidad "
       "perfecta aísla el precio de la estocasticidad."),
])


# --------------------------------------------------------------------------- #
# 02 — Metaheurísticas
# --------------------------------------------------------------------------- #

nb02 = notebook([
    header("Pipeline EHBG-FACS · 02 · Metaheurísticas (ACO y Tabu)",
           "Paradigma 2 — implementaciones oficiales de SVRPBench, re-puntuadas con CRN.",
           "Envuelve el **Ant System** y la **Tabu Search** oficiales del benchmark (con arranque "
           "NN+2opt) y re-puntúa sus rutas con el evaluador compartido, para que sean comparables "
           "con el resto. Agregación best-of-K (multistart) determinista por instancia."),
    code(SETUP_CODE),
    code(CONFIG_CODE),
    md("## Resolver ACO y Tabu\n`n_jobs=N_JOBS` reparte las instancias entre procesos fork "
       "(los solvers heredados son Python puro: el GIL impide usar hilos), y las K corridas "
       "best-of-K de cada instancia se puntúan **compartiendo el muestreo ξ** "
       "(`score_routes_multi`: idéntico bit a bit, muestreo ÷K). **Nota de escala:** "
       "a n≥200 el costo de construcción es O(iteraciones·hormigas·n²); con las 2 vCPU del "
       "runtime T4 espera decenas de minutos — un runtime A100/L4 (8–12 vCPU) lo reduce ~5×."),
    code('from svrplab.solvers.metaheuristic import ACO, Tabu\n'
         'import pandas as pd\n'
         'df_aco  = runner.run_solver(ACO(n_seeds=5),  "aco",  bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS)\n'
         'df_tabu = runner.run_solver(Tabu(n_seeds=5), "tabu", bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS)\n'
         'df = pd.concat([df_aco, df_tabu], ignore_index=True)\n'
         'df'),
    md("## Métricas y figuras\nExportadas a `figures/02_metaheuristic/`."),
    code('display(metrics.aggregate_by_size(df))\n'
         'fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "02_metaheuristic", "metricas_por_tamano")\n'
         'inst = bank[SIZES[0]][0]\n'
         'sol = ACO(n_seeds=5).solve(inst, num_realizations=proto.realizations)\n'
         'fig = viz.plot_routes(inst, sol.routes, title=f"aco · n={SIZES[0]}")\n'
         'viz.save_show(fig, env, "02_metaheuristic", f"rutas_aco_n{SIZES[0]}")'),
    md("**Interpretación.** Las metaheurísticas alcanzan **factibilidad alta** pero usando "
       "**más vehículos** (rutas cortas) y, por tanto, mayor costo: la otra familia del tradeoff."),
])


# --------------------------------------------------------------------------- #
# 03 — NCO supervisado
# --------------------------------------------------------------------------- #

nb03 = notebook([
    header("Pipeline EHBG-FACS · 03 · NCO supervisado (Attention Model)",
           "Paradigma 3 — Attention Model entrenado por imitación de un maestro.",
           "Un **Transformer codificador-decodificador** (Kool/Kwon) entrenado por *teacher "
           "forcing* para imitar rutas etiquetadas por un **maestro** (caro): `exact-bc` (óptimo, "
           "ignora ventanas) en `nco-sl`, o `aco` (factible) en `nco-sl-feas`. La (in)factibilidad "
           "la define el maestro. Inferencia en milisegundos; entrenamiento amortizado en GPU."),
    code(SETUP_CODE),
    code(CONFIG_CODE),
    md("## Entrenar e inferir (GPU)\nLas etiquetas se generan resolviendo instancias de "
       "entrenamiento con el maestro (requiere Gurobi si el maestro es `exact-bc`); esa fase "
       "es CPU y se reparte entre `N_JOBS` procesos. El modelo se cachea en Drive. Ajusta "
       "`epochs`/`n_per_size` según el tiempo disponible."),
    code('# Maestro exact-bc requiere Gurobi:\n'
         'import subprocess, sys; subprocess.run([sys.executable,"-m","pip","install","-q","gurobipy"], check=False)\n'
         'from svrplab.solvers.nco_sl import NCOSupervised, NCOSupervisedFeasible\n'
         'import pandas as pd\n'
         'common = dict(train_sizes=(10,20), n_per_size=256, epochs=80, embed_dim=128,\n'
         '              device=env.device, models_dir=env.paths.models, n_jobs=N_JOBS, verbose=True)\n'
         'sl      = NCOSupervised(teacher="exact-bc", **common)\n'
         'sl_feas = NCOSupervisedFeasible(**common)   # maestro = aco (factible)\n'
         'df_sl   = runner.run_solver(sl,      "nco-sl",      bank, env, proto, verbose=True, n_jobs=N_JOBS)\n'
         'df_feas = runner.run_solver(sl_feas, "nco-sl-feas", bank, env, proto, verbose=True, n_jobs=N_JOBS)\n'
         'df = pd.concat([df_sl, df_feas], ignore_index=True); df'),
    md("## Curva de entrenamiento y figuras\nExportadas a `figures/03_nco_supervised/`. "
       "**Limitación declarada:** el modelo se entrena en n∈{10,20}; evaluar en n≥100 es "
       "generalización fuera de distribución (con más cómputo, añade 50/100 a `train_sizes`)."),
    code('if getattr(sl, "history", None):\n'
         '    fig = viz.plot_training_curve(sl.history, ylabel="CE", title="nco-sl: pérdida de imitación")\n'
         '    viz.save_show(fig, env, "03_nco_supervised", "curva_entrenamiento_nco_sl")\n'
         'display(metrics.aggregate_by_size(df))\n'
         'fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "03_nco_supervised", "metricas_por_tamano")'),
    md("**Interpretación.** `nco-sl` (imita al óptimo) hereda baja factibilidad; `nco-sl-feas` "
       "(imita a aco) hereda factibilidad alta — la imitación voraz es **con pérdida**. Confirma "
       "que el límite proviene del **maestro**, no del paradigma NCO."),
])


# --------------------------------------------------------------------------- #
# 04 — NCO por RL (POMO)
# --------------------------------------------------------------------------- #

nb04 = notebook([
    header("Pipeline EHBG-FACS · 04 · NCO por RL (POMO + Attention Model)",
           "Paradigma 4 — Attention Model entrenado por REINFORCE estilo POMO (GPU).",
           "Política neuronal entrenada **sin etiquetas**, con el costo de la ruta como recompensa "
           "y la estrategia **POMO**: N trayectorias desde nodos de inicio distintos + media como "
           "**línea base compartida**. Recompensa = costo **determinista** (tiempo nominal τ) → por "
           "eso es 'NCO determinista' y, evaluada bajo ξ, exhibe fragilidad ante la estocasticidad."),
    code(SETUP_CODE),
    code(CONFIG_CODE),
    md("## Entrenar (POMO, GPU) e inferir\nEntrenamiento autoregresivo con bonus de entropía y "
       "AMP en GPU. Aumenta `steps_per_size` para mayor calidad (más exigente en cómputo). "
       "La inferencia corre en GPU; la re-puntuación CRN (CPU) se paraleliza con `n_jobs`. "
       "**Recompensa consciente de ventanas** (`tw_penalty`): costo nominal + penalización de "
       "tardanza nominal — la política sigue siendo *determinista* (no ve ξ) pero **busca "
       "satisfacer las restricciones** en vez de ignorarlas (pon `tw_penalty=0` para recuperar "
       "el baseline puro de costo). **Limitación declarada:** entrenado en n∈{10,20} — en n≥100 "
       "es generalización fuera de distribución (la literatura POMO entrena n=100; súbelo si "
       "tienes A100)."),
    code('from svrplab.solvers.nco_rl import NCOReinforce\n'
         'rl = NCOReinforce(train_sizes=(10,20), steps_per_size=1500, batch=64, embed_dim=128,\n'
         '                  tw_penalty=proto.late_penalty,   # 0 = baseline puro de costo\n'
         '                  device=env.device, models_dir=env.paths.models, verbose=True)\n'
         'df = runner.run_solver(rl, "nco-rl", bank, env, proto, verbose=True, n_jobs=N_JOBS)\n'
         'df'),
    md("## Curva de entrenamiento y figuras\nExportadas a `figures/04_nco_pomo_am/`."),
    code('if getattr(rl, "history", None):\n'
         '    fig = viz.plot_training_curve(rl.history, ylabel="costo medio", title="nco-rl (POMO): costo")\n'
         '    viz.save_show(fig, env, "04_nco_pomo_am", "curva_entrenamiento_pomo")\n'
         'display(metrics.aggregate_by_size(df))\n'
         'inst = bank[SIZES[0]][0]\n'
         'sol = rl.solve(inst, num_realizations=proto.realizations)\n'
         'fig = viz.plot_routes(inst, sol.routes, title=f"nco-rl (POMO) · n={SIZES[0]}")\n'
         'viz.save_show(fig, env, "04_nco_pomo_am", f"rutas_pomo_n{SIZES[0]}")'),
    md("**Interpretación.** Bien entrenado, POMO se acerca al óptimo de costo y **supera a la NCO "
       "supervisada** (coherente con la literatura: RL > supervisado). Con `tw_penalty>0` la "
       "política busca cumplir las ventanas en el plan nominal; la degradación restante bajo ξ "
       "es atribuible a la **estocasticidad** (retrasos log-normales, accidentes), no a ignorar "
       "las restricciones — exactamente la fragilidad que el anteproyecto atribuye a la NCO "
       "determinista."),
])


# --------------------------------------------------------------------------- #
# 05 — EHBG-FACS (propuesta)
# --------------------------------------------------------------------------- #

nb05 = notebook([
    header("Pipeline EHBG-FACS · 05 · Propuesta (HBG-GFlowNet + GFACS + ENN)",
           "Paradigma 5 — muestreo híbrido GFlowNet de Balance Híbrido + colonia de hormigas.",
           "La propuesta de la tesis. Una **GFlowNet de Balance Híbrido (HBG)** —Attention Model "
           "que parametriza P_F, P_B y el flujo F_θ(s), entrenada con el objetivo híbrido **TB+DB** "
           "(ponderado por λ_DB) y **recompensa sensible al riesgo** R(x) ∝ exp(−CVaR_α/T)— acopla "
           "su matriz heurística a un **muestreador de Colonia de Hormigas (GFACS)** con búfer de "
           "repetición fuera de política. Busca simultáneamente **bajo costo y alta factibilidad** "
           "(la región que ningún paradigma base ocupa). La variante **ENN** activa la cabeza "
           "epinet para guiar la exploración con incertidumbre epistémica (Fase 5)."),
    code(SETUP_CODE),
    code(CONFIG_CODE),
    md("## Entrenar EHBG-FACS (GPU)\nEntrena la GFlowNet HBG con recompensa CVaR + refinamiento "
       "GFACS + replay off-policy, sobre un banco de entrenamiento fijo (semillas disjuntas del de "
       "evaluación). Hiperparámetros clave (Cuadro 2 del anteproyecto): `lam_db` (TB↔DB), "
       "`temperature` (suavizado de la política), `rho` (evaporación ACO). `n_jobs=N_JOBS` "
       "paraleliza en procesos las fases CPU: puntuación CRN del lote, refinamiento GFACS por "
       "instancia y el **enjambre de hormigas** en inferencia (los escenarios ξ se pre-muestrean "
       "una sola vez por instancia y se comparten entre hormigas — mismo resultado, mucho más "
       "rápido a n grande)."),
    code('from svrplab.solvers.ehbg_facs import EHBGFACS\n'
         'facs = EHBGFACS(train_sizes=(10,20), n_train=64, epochs=40, embed_dim=128,\n'
         '                lam_db=0.5, temperature=2.0, batch=16, refine_every=5,\n'
         '                infer_ants=16, infer_iters=12, infer_realizations=40,\n'
         '                device=env.device, models_dir=env.paths.models, n_jobs=N_JOBS,\n'
         '                verbose=True)\n'
         'df = runner.run_solver(facs, "ehbg-facs", bank, env, proto, verbose=True, n_jobs=N_JOBS)\n'
         'df'),
    md("## (Opcional, Fase 5) Variante epistémica EHBG-FACS-ENN"),
    code('from svrplab.solvers.ehbg_facs import EHBGFACSEpistemic\n'
         'facs_enn = EHBGFACSEpistemic(train_sizes=(10,20), n_train=64, epochs=40, embed_dim=128,\n'
         '                             lam_db=0.5, temperature=2.0, batch=16, refine_every=5,\n'
         '                             infer_ants=16, infer_iters=12, infer_realizations=40,\n'
         '                             device=env.device, models_dir=env.paths.models,\n'
         '                             n_jobs=N_JOBS, verbose=True)\n'
         'df_enn = runner.run_solver(facs_enn, "ehbg-facs-enn", bank, env, proto, verbose=True,\n'
         '                           n_jobs=N_JOBS)\n'
         'df_enn'),
    md("## Curvas de entrenamiento (TB / DB / CVaR) y figuras\nExportadas a `figures/05_ehbg_facs/`."),
    code('import matplotlib.pyplot as plt\n'
         'h = getattr(facs, "history", {})\n'
         'if h:\n'
         '    fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))\n'
         '    ax[0].plot(h["tb"]); ax[0].set_title("Balance de Trayectoria (TB)")\n'
         '    ax[1].plot(h["db"], c="orange"); ax[1].set_title("Balance Detallado (DB)")\n'
         '    ax[2].plot(h["cvar"], c="crimson"); ax[2].set_title("CVaR medio (recompensa)")\n'
         '    for a in ax: a.set_xlabel("paso")\n'
         '    plt.tight_layout()\n'
         '    viz.save_show(fig, env, "05_ehbg_facs", "curvas_tb_db_cvar")\n'
         'display(metrics.aggregate_by_size(df))\n'
         'inst = bank[SIZES[0]][0]\n'
         'sol = facs.solve(inst, num_realizations=proto.realizations)\n'
         'fig = viz.plot_routes(inst, sol.routes, title=f"EHBG-FACS · n={SIZES[0]}")\n'
         'viz.save_show(fig, env, "05_ehbg_facs", f"rutas_ehbg_facs_n{SIZES[0]}")'),
    md("**Interpretación.** EHBG-FACS combina el muestreo adaptativo (diversidad de la GFlowNet) "
       "con el refinamiento poblacional del ACO y una recompensa de cola (CVaR), apuntando a la "
       "**esquina ideal** (bajo costo *y* alta factibilidad) que el informe técnico identificó "
       "vacía. Compáralo con los baselines en el notebook 06."),
])


# --------------------------------------------------------------------------- #
# 06 — Comparación y estadística
# --------------------------------------------------------------------------- #

nb06 = notebook([
    header("Pipeline EHBG-FACS · 06 · Comparación y validación estadística",
           "Reúne los resultados de los 5 paradigmas y aplica ANOVA / Friedman / Wilcoxon.",
           "Carga todos los `*_metrics.csv` escritos por los notebooks 01–05 (misma raíz de Drive), "
           "construye la tabla comparativa y el tradeoff costo–factibilidad–flota, y contrasta las "
           "diferencias con **rigor estadístico** (Fase 4 del anteproyecto): verificación de "
           "supuestos (Shapiro, Levene), ANOVA cuando se cumplen y Friedman + post-hoc Wilcoxon "
           "(corrección de Holm) cuando no, dada la cola pesada de los costos."),
    code(SETUP_CODE),
    code(CONFIG_CODE),
    md("## Cargar todos los resultados\nSe **filtra** al banco de la configuración actual "
       "(`SIZES` × `instance < N_INSTANCES`): así una corrida vieja con otra configuración que "
       "siga en Drive no contamina la comparación. La auditoría de cobertura muestra qué solver "
       "corrió qué tamaños (los bloques incompletos se excluyen automáticamente en las pruebas "
       "pareadas)."),
    code('df = runner.load_all_results(env)\n'
         'antes = len(df)\n'
         'df = df[df["size"].isin(SIZES) & (df["instance"] < N_INSTANCES)].copy()\n'
         'print(f"filas: {len(df)} (descartadas {antes - len(df)} de corridas con otra configuración)")\n'
         'print("solvers:", sorted(df.solver.unique()))\n'
         'cobertura = df.pivot_table(index="solver", columns="size", values="instance",\n'
         '                           aggfunc="count").fillna(0).astype(int)\n'
         'display(cobertura)   # instancias por (solver, tamaño); 0 = no corrió ese tamaño'),
    md("## Tabla resumen (leaderboard)\nPromedio sobre todo el banco; ordenado por costo total "
       "esperado con recurso."),
    code('display(metrics.leaderboard(df, by="expected_total"))\n'
         'display(metrics.aggregate_by_size(df))'),
    md("## Figuras comparativas\nBarras por tamaño y tradeoff costo–factibilidad–flota (la región "
       "ideal es arriba-izquierda: bajo costo y alta factibilidad). Todas quedan en "
       "`figures/cross/`."),
    code('fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "cross", "comparison_metrics")\n'
         'for s in sorted(df["size"].unique()):\n'
         '    fig = viz.plot_tradeoff(df, size=int(s))\n'
         '    viz.save_show(fig, env, "cross", f"tradeoff_n{int(s)}")'),
    md("## Validación estadística\nPara cada métrica clave y cada tamaño: supuestos, prueba "
       "ómnibus (ANOVA/Friedman) y post-hoc Wilcoxon pareado (Holm). Diseño de **bloques por "
       "instancia** (mismo ξ por CRN). **Potencia:** con 5 bloques el p bilateral mínimo de "
       "Wilcoxon es 0.0625 — imposible declarar significancia a α=0.05; estos resultados son "
       "exploratorios hasta correr con N_INSTANCES=30."),
    code('if df.groupby(["solver","size"])["instance"].count().min() < 10:\n'
         '    print("ADVERTENCIA: <10 bloques por celda -> pruebas SIN potencia; corrida exploratoria.\\n")\n'
         'for metric in ["expected_total", "cvar", "feasibility"]:\n'
         '    for s in sorted(df["size"].unique()):\n'
         '        cmp = metrics.compare_solvers(df, metric=metric, size=int(s), alpha=proto.significance)\n'
         '        print("="*70)\n'
         '        print(metrics.summarize_comparison(cmp))'),
    md("## Guardar resumen\nEscribe la tabla, el resumen estadístico (JSON + texto legible) en "
       "`results/cross/`."),
    code('import json, pandas as pd\n'
         'out = env.paths.results / "cross"; out.mkdir(parents=True, exist_ok=True)\n'
         'metrics.leaderboard(df).to_csv(out / "leaderboard.csv", index=False)\n'
         'metrics.aggregate_by_size(df).to_csv(out / "aggregate_by_size.csv", index=False)\n'
         'stats = {f"{m}_n{int(s)}": metrics.compare_solvers(df, metric=m, size=int(s))\n'
         '         for m in ["expected_total","cvar","feasibility"] for s in sorted(df["size"].unique())}\n'
         '(out / "statistics.json").write_text(json.dumps(stats, indent=2, default=float))\n'
         'texto = "\\n\\n".join(metrics.summarize_comparison(c) for c in stats.values())\n'
         '(out / "statistics.txt").write_text(texto)\n'
         'print("guardado en", out)'),
    md("**Lectura final.** Si EHBG-FACS se ubica en la región ideal (bajo `E[c]`/`CVaR` con "
       "`feasibility` alta) y la diferencia frente a los baselines es **estadísticamente "
       "significativa** (p_holm < α) en costo/CVaR a factibilidad comparable, se sostiene la "
       "hipótesis general del anteproyecto."),
])


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "00_setup_y_datos.ipynb": nb00,
        "01_exactos_branch_cut.ipynb": nb01,
        "02_metaheuristicas_aco_tabu.ipynb": nb02,
        "03_nco_supervisado_attention.ipynb": nb03,
        "04_nco_rl_pomo.ipynb": nb04,
        "05_ehbg_facs.ipynb": nb05,
        "06_comparacion_y_estadistica.ipynb": nb06,
    }
    for name, nb in files.items():
        (NB_DIR / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        print("escrito", NB_DIR / name)


if __name__ == "__main__":
    main()
