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

# Estudio de caso COMÚN: la misma instancia se dibuja paso a paso en TODOS los
# notebooks (comparación visual justa entre métodos). n=20 garantiza que hasta
# los exactos con licencia restringida la cubren.
CASE_SIZE, CASE_IDX = 20, 0

bank = data.load_bank(env.paths.instances, SIZES, N_INSTANCES,
                      base_seed=proto.base_seed, capacity_mode=proto.capacity_mode, verbose=True)
print({s: len(v) for s, v in bank.items()}, "instancias por tamaño")
print("huella del banco (auditoría de piso parejo):", data.bank_fingerprint(bank))
if N_INSTANCES < 10:
    print("ADVERTENCIA: con N_INSTANCES=5 el Wilcoxon pareado NO puede alcanzar p<0.05 "
          "(mínimo teórico bilateral con n=5: 0.0625). Es una corrida exploratoria; "
          "para las conclusiones de la tesis usa N_INSTANCES=30.")
'''.strip("\n")

# Celda reutilizable de estudio de caso: misma instancia + misma representación
# (progresión del recorrido) en todos los notebooks → comparación visual justa.
def case_study_code(slug, solver_var, name):
    return code(
        f'# === Estudio de caso: instancia común n=CASE_SIZE, idx=CASE_IDX ============\n'
        f'inst_c = bank[CASE_SIZE][CASE_IDX]\n'
        f'sol_c  = {solver_var}.solve(inst_c, num_realizations=proto.realizations)\n'
        f'fig = viz.plot_routes(inst_c, sol_c.routes,\n'
        f'                      title=f"{name} · caso n={{CASE_SIZE}} inst={{CASE_IDX}}")\n'
        f'viz.save_show(fig, env, "{slug}", f"caso_n{{CASE_SIZE}}_i{{CASE_IDX}}_rutas_{name}")\n'
        f'fig = viz.plot_route_progression(inst_c, sol_c.routes, n_frames=6,\n'
        f'                                 title=f"{name}: progresión de la solución")\n'
        f'viz.save_show(fig, env, "{slug}", f"caso_n{{CASE_SIZE}}_i{{CASE_IDX}}_progresion_{name}")')


CASE_MD = md("## Estudio de caso (misma instancia en todos los notebooks)\n"
             "Se resuelve y exporta paso a paso la **misma** instancia "
             "(`CASE_SIZE`, `CASE_IDX`, fija en la configuración) con la **misma** "
             "representación visual (progresión del recorrido): el notebook 06 "
             "reúne estas figuras en una rejilla método-a-método. Así la "
             "comparación visual es justa — se compara la solución, no el estilo "
             "del dibujo.")


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
    md("## 6. Glosario de métricas (qué significa cada columna del CSV)\n"
       "El núcleo de la suite reproduce las métricas del paper de SVRPBench (Heakl et al. "
       "2025, §4.1): **TC** = `expected_cost` (Eq. 15), **CVR** = `cvr` (Eq. 16), **FR** = "
       "`feasibility` (Eq. 17), **RT** = `runtime`, **ROB** = `rob_var` (Eq. 18, varianza) y "
       "`waiting_time` (Fig. 4). Las demás columnas son **extensiones declaradas** de esta "
       "tesis (recurso de 2ª etapa y riesgo: `E[c+Q]`, `E[Q]`, `CVaR/VaR`, extremos y "
       "dispersión de `c+Q`). Diferencia de protocolo declarada: el paper promedia sobre 5 "
       "realizaciones; aquí usamos `realizations=200` con CRN — necesario para estimar la "
       "cola (CVaR al 95% con 5 muestras no es estimable). El glosario se exporta junto a "
       "los resultados."),
    code('glos = metrics.metrics_glossary()\n'
         'display(glos)\n'
         'glos.to_csv(env.paths.results / "metrics_glossary.csv", index=False)\n'
         'print("glosario ->", env.paths.results / "metrics_glossary.csv")'),
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
         '                          n_jobs=N_JOBS, cost_samples=True)\n'
         'df_bc'),
    md("## Resolver `exact-bc-tw` (CVRPTW soft — **busca satisfacer las ventanas**)\n"
       "Formulación dirigida MTZ que penaliza la tardanza nominal `L_j` en el objetivo "
       "(ventanas *soft*: por el caveat de escala de SVRPBench las ventanas duras son casi "
       "insatisfacibles desde n≈20). Aísla el efecto de la estocasticidad: la infactibilidad "
       "residual bajo ξ es atribuible a la incertidumbre, no a ignorar las restricciones."),
    code('solver_tw = ExactBranchCutTW(time_limit=120.0, tw_penalty=proto.late_penalty,\n'
         '                             verbose=False)\n'
         'df_tw = runner.run_solver(solver_tw, "exact-bc-tw", bank_tw, env, proto, verbose=True,\n'
         '                          n_jobs=N_JOBS, cost_samples=True)\n'
         'df = pd.concat([df_bc, df_tw], ignore_index=True)\n'
         'df_tw'),
    md("## Métricas agregadas y figuras\nTodas las figuras quedan en `figures/01_exact/`."),
    code('agg = metrics.aggregate_by_size(df); display(agg)\n'
         'fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "01_exact", "metricas_por_tamano")'),
    CASE_MD,
    case_study_code("01_exact", "solver", "exact-bc"),
    code('# La búsqueda del exacto se documenta con su convergencia B&C (incumbente vs. cota)\n'
         'fig = viz.plot_convergence(sol_c.extras.get("convergence_log", []),\n'
         '                           gap=sol_c.extras.get("gap"), n=CASE_SIZE)\n'
         'viz.save_show(fig, env, "01_exact", f"caso_n{CASE_SIZE}_i{CASE_IDX}_convergencia_bc")'),
    case_study_code("01_exact", "solver_tw", "exact-bc-tw"),
    md("**Qué observar (guía de lectura — las conclusiones se toman en el notebook 06 con "
       "estadística, no aquí).** (i) ¿Difieren `E[c]` y `E[c+Q]` entre `exact-bc` y "
       "`exact-bc-tw`, y en qué dirección? (ii) ¿Cuánta tardanza nominal "
       "(`nominal_tw_lateness` en extras) elimina la variante TW y a qué precio en costo de "
       "viaje? (iii) `gap`: ¿en qué tamaños el B&C prueba optimalidad dentro del límite de "
       "tiempo y dónde empieza a agotar el presupuesto? (iv) `fallback`: debe ser 0 en "
       "condiciones normales."),
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
         'aco_solver, tabu_solver = ACO(n_seeds=5), Tabu(n_seeds=5)\n'
         'df_aco  = runner.run_solver(aco_solver,  "aco",  bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS, cost_samples=True)\n'
         'df_tabu = runner.run_solver(tabu_solver, "tabu", bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS, cost_samples=True)\n'
         'df = pd.concat([df_aco, df_tabu], ignore_index=True)\n'
         'df'),
    md("## Métricas y figuras\nExportadas a `figures/02_metaheuristic/`."),
    code('display(metrics.aggregate_by_size(df))\n'
         'fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "02_metaheuristic", "metricas_por_tamano")'),
    CASE_MD,
    case_study_code("02_metaheuristic", "aco_solver", "aco"),
    case_study_code("02_metaheuristic", "tabu_solver", "tabu"),
    md("**Qué observar (guía de lectura — las conclusiones se toman en el notebook 06 con "
       "estadística, no aquí).** (i) Factibilidad y CVR frente al número de vehículos: ¿cómo "
       "negocia cada metaheurística el tradeoff flota↔cumplimiento? (ii) `seed_std_cost` "
       "(extras): ¿cuánta variabilidad hay entre las semillas del multistart best-of-K? "
       "(iii) `runtime`: ¿cómo escala el costo de construcción con n?"),
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
         'df_sl   = runner.run_solver(sl,      "nco-sl",      bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS, cost_samples=True)\n'
         'df_feas = runner.run_solver(sl_feas, "nco-sl-feas", bank, env, proto, verbose=True,\n'
         '                            n_jobs=N_JOBS, cost_samples=True)\n'
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
    CASE_MD,
    case_study_code("03_nco_supervised", "sl", "nco-sl"),
    case_study_code("03_nco_supervised", "sl_feas", "nco-sl-feas"),
    md("**Qué observar (guía de lectura — conclusiones en el notebook 06).** (i) ¿Difieren "
       "factibilidad y `E[c+Q]` entre `nco-sl` (maestro exact-bc) y `nco-sl-feas` (maestro "
       "aco)? El diseño permite atribuir esa diferencia al **maestro**, no al paradigma. "
       "(ii) `runtime` de inferencia frente a los métodos que buscan por instancia. "
       "(iii) CE de validación: ¿converge la imitación sin sobreajuste?"),
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
         'df = runner.run_solver(rl, "nco-rl", bank, env, proto, verbose=True, n_jobs=N_JOBS,\n'
         '                       cost_samples=True)\n'
         'df'),
    md("## Curva de entrenamiento y figuras\nExportadas a `figures/04_nco_pomo_am/`."),
    code('if getattr(rl, "history", None):\n'
         '    fig = viz.plot_training_curve(rl.history, ylabel="costo medio", title="nco-rl (POMO): costo")\n'
         '    viz.save_show(fig, env, "04_nco_pomo_am", "curva_entrenamiento_pomo")\n'
         'display(metrics.aggregate_by_size(df))'),
    CASE_MD,
    case_study_code("04_nco_pomo_am", "rl", "nco-rl"),
    md("**Qué observar (guía de lectura — conclusiones en el notebook 06).** (i) ¿Se degradan "
       "factibilidad/CVaR al pasar del plan nominal a la evaluación bajo ξ, y cuánto? Con "
       "`tw_penalty>0` esa degradación es atribuible a la estocasticidad, no a ignorar "
       "restricciones. (ii) ¿La curva de entrenamiento converge de forma estable (sin colapso "
       "de modo)? (iii) `runtime` de inferencia frente a los métodos de búsqueda por instancia. "
       "(iv) En n≥100, recuerda que el modelo opera fuera de su distribución de entrenamiento."),
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
         'df = runner.run_solver(facs, "ehbg-facs", bank, env, proto, verbose=True, n_jobs=N_JOBS,\n'
         '                       cost_samples=True)\n'
         'df'),
    md("## (Opcional, Fase 5) Variante epistémica EHBG-FACS-ENN"),
    code('from svrplab.solvers.ehbg_facs import EHBGFACSEpistemic\n'
         'facs_enn = EHBGFACSEpistemic(train_sizes=(10,20), n_train=64, epochs=40, embed_dim=128,\n'
         '                             lam_db=0.5, temperature=2.0, batch=16, refine_every=5,\n'
         '                             infer_ants=16, infer_iters=12, infer_realizations=40,\n'
         '                             device=env.device, models_dir=env.paths.models,\n'
         '                             n_jobs=N_JOBS, verbose=True)\n'
         'df_enn = runner.run_solver(facs_enn, "ehbg-facs-enn", bank, env, proto, verbose=True,\n'
         '                           n_jobs=N_JOBS, cost_samples=True)\n'
         'df_enn'),
    md("## Curvas de entrenamiento (TB / DB / CVaR)\nDiagnósticos **propios de la GFlowNet** "
       "(H1 del anteproyecto): la pérdida TB documenta el balance global de trayectoria, la DB "
       "la consistencia local, y el CVaR medio la señal de recompensa. Exportadas a "
       "`figures/05_ehbg_facs/`."),
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
         'display(metrics.aggregate_by_size(df))'),
    CASE_MD,
    case_study_code("05_ehbg_facs", "facs", "ehbg-facs"),
    md("## Diagnósticos del muestreo (traza GFACS, diversidad y distribución de costo)\n"
       "Evidencia **medible** de los mecanismos de la propuesta sobre el estudio de caso "
       "(H2/H3 del anteproyecto): (a) traza de búsqueda del enjambre (mejor CVaR por "
       "iteración); (b) **diversidad muestral** = proporción de soluciones distintas entre "
       "hormigas por iteración (diagnóstico anti-colapso de modo; queda también en la columna "
       "`diversity` del CSV); (c) distribución completa del costo `c+Q` bajo los 200 "
       "escenarios ξ con su CVaR — la métrica que la recompensa optimiza. Ningún baseline "
       "expone estos diagnósticos porque no muestrea una distribución; la comparación de "
       "resultados finales sigue siendo con las métricas comunes del notebook 06."),
    code('tr = sol_c.extras.get("search_trace", {})\n'
         'if tr.get("best_cvar"):\n'
         '    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))\n'
         '    ax[0].plot(tr["iter_best_cvar"], "o-", label="mejor de la iteración")\n'
         '    ax[0].plot(tr["best_cvar"], "-", c="crimson", label="mejor global")\n'
         '    ax[0].set_xlabel("iteración GFACS"); ax[0].set_ylabel("CVaR"); ax[0].legend()\n'
         '    ax[0].set_title("Traza de búsqueda del enjambre")\n'
         '    ax[1].plot(tr["unique_ratio"], "s-", c="teal"); ax[1].set_ylim(0, 1.05)\n'
         '    ax[1].set_xlabel("iteración GFACS")\n'
         '    ax[1].set_title("Diversidad muestral (soluciones únicas / hormigas)")\n'
         '    plt.tight_layout()\n'
         '    viz.save_show(fig, env, "05_ehbg_facs", f"caso_n{CASE_SIZE}_i{CASE_IDX}_traza_diversidad")\n'
         'print(f"diversidad media del muestreo: {sol_c.extras.get(\'diversity\', float(\'nan\')):.2f}")\n'
         'from svrplab import stochastic\n'
         'sc = stochastic.score_routes(inst_c, sol_c.routes, num_realizations=proto.realizations,\n'
         '                             seed=int(inst_c.metadata["seed"]), alpha=proto.alpha,\n'
         '                             late_penalty=proto.late_penalty,\n'
         '                             accident_scale=proto.accident_scale)\n'
         'fig = viz.plot_cost_hist(sc.total_samples, alpha=proto.alpha,\n'
         '                         title=f"EHBG-FACS: distribución de c+Q bajo ξ (caso n={CASE_SIZE})")\n'
         'viz.save_show(fig, env, "05_ehbg_facs", f"caso_n{CASE_SIZE}_i{CASE_IDX}_dist_costo")'),
    md("**Qué observar (guía de lectura — conclusiones en el notebook 06).** (i) ¿Descienden "
       "TB/DB de forma estable (H1)? (ii) ¿Se mantiene la diversidad muestral alta a lo largo "
       "de las iteraciones GFACS, o colapsa (H2)? (iii) ¿La traza del enjambre sigue mejorando "
       "el CVaR en las últimas iteraciones (presupuesto bien usado) o se satura pronto? "
       "(iv) En la distribución de c+Q: ¿la cola derecha es corta respecto a los baselines "
       "(compárala en el notebook 06 bajo los MISMOS ξ)? (v) `diversity` en el CSV frente a "
       "la factibilidad/CVaR: ¿coexisten diversidad y calidad?"),
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
    md("## Auditoría del piso parejo\nCada corrida persiste en su `run.json` la **huella por "
       "tamaño** del banco que resolvió (hash de las semillas de sus instancias). Aquí se "
       "compara contra el banco actual: cualquier discrepancia significa que ese solver corrió "
       "sobre OTRAS instancias y su comparación no sería válida."),
    code('import pandas as pd\n'
         'ref = data.size_fingerprints(bank)\n'
         'meta = runner.load_run_meta(env)\n'
         'filas = []\n'
         'for sv, m in sorted(meta.items()):\n'
         '    sfp = {int(k): v for k, v in (m.get("size_fingerprints") or {}).items()}\n'
         '    difieren = [s for s, h in sfp.items() if s in ref and ref[s] != h]\n'
         '    filas.append({"solver": sv, "tamaños": sorted(sfp), "device": m.get("device"),\n'
         '                  "realizations": (m.get("protocol") or {}).get("realizations"),\n'
         '                  "banco_ok": not difieren, "tamaños_discrepantes": difieren})\n'
         'audit = pd.DataFrame(filas)\n'
         'display(audit)\n'
         'assert audit["banco_ok"].all(), "PISO PAREJO ROTO: hay solvers con instancias distintas"\n'
         'print("Piso parejo verificado: todos los solvers resolvieron las mismas instancias "\n'
         '      "en los tamaños que cubren.")'),
    md("## Tabla resumen (leaderboard)\nPromedio sobre todo el banco; ordenado por costo total "
       "esperado con recurso. El glosario (también en `results/metrics_glossary.csv`) define "
       "cada columna, sus unidades, la dirección deseable y si proviene del paper de SVRPBench "
       "(Eqs. 15–18) o es una extensión declarada."),
    code('display(metrics.metrics_glossary())\n'
         'display(metrics.leaderboard(df, by="expected_total"))\n'
         'display(metrics.aggregate_by_size(df))'),
    md("## Figuras comparativas\nBarras por tamaño y tradeoff costo–factibilidad–flota (la región "
       "ideal es arriba-izquierda: bajo costo y alta factibilidad). Todas quedan en "
       "`figures/cross/`."),
    code('fig = viz.plot_comparison(df)\n'
         'viz.save_show(fig, env, "cross", "comparison_metrics")\n'
         'for s in sorted(df["size"].unique()):\n'
         '    fig = viz.plot_tradeoff(df, size=int(s))\n'
         '    viz.save_show(fig, env, "cross", f"tradeoff_n{int(s)}")'),
    md("## Estudio de caso: la misma instancia, método a método\nCada notebook persiste sus "
       "rutas (`*_routes.json`) y sus muestras de costo (`*_samples.npz`). Aquí se dibuja la "
       "**misma instancia** (`CASE_SIZE`, `CASE_IDX`) resuelta por cada método (rejilla, con "
       "E[c+Q] y factibilidad anotadas) y se superponen las **distribuciones del costo c+Q "
       "bajo los MISMOS 200 escenarios ξ** (CRN): las diferencias entre curvas son atribuibles "
       "únicamente a las rutas, no al azar de la simulación."),
    code('inst_c = bank[CASE_SIZE][CASE_IDX]\n'
         'key = f"{CASE_SIZE}:{CASE_IDX}"\n'
         'rutas, mets, dists = {}, {}, {}\n'
         'for sv in sorted(df.solver.unique()):\n'
         '    r = runner.load_routes(env, sv).get(key)\n'
         '    if not r:\n'
         '        continue    # ese solver no cubrió el tamaño del caso (p. ej. licencia)\n'
         '    rutas[sv] = r\n'
         '    fila = df[(df.solver == sv) & (df["size"] == CASE_SIZE) & (df.instance == CASE_IDX)]\n'
         '    if len(fila):\n'
         '        mets[sv] = fila.iloc[0].to_dict()\n'
         '    s = runner.load_samples(env, sv).get(key)\n'
         '    if s is not None:\n'
         '        dists[sv] = s\n'
         'print("métodos con el caso resuelto:", sorted(rutas))\n'
         'fig = viz.plot_case_grid(inst_c, rutas, metrics_by_solver=mets,\n'
         '                         title=f"Estudio de caso · n={CASE_SIZE}, instancia {CASE_IDX}")\n'
         'viz.save_show(fig, env, "cross", f"caso_n{CASE_SIZE}_i{CASE_IDX}_grid")\n'
         'if dists:\n'
         '    fig = viz.plot_cost_distributions(dists, alpha=proto.alpha,\n'
         '        title=f"c+Q bajo los MISMOS xi (CRN) · n={CASE_SIZE} inst {CASE_IDX}")\n'
         '    viz.save_show(fig, env, "cross", f"caso_n{CASE_SIZE}_i{CASE_IDX}_distribuciones")'),
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
