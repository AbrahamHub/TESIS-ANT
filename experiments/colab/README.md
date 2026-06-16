# Pipeline EHBG-FACS sobre SVRPBench (Google Colab / GPU)

Pipeline de experimentación reproducible para comparar, con **piso parejo** y **rigor
científico**, los cinco paradigmas del anteproyecto de tesis sobre el **Problema de
Enrutamiento de Vehículos Estocástico (SVRP)**, usando el benchmark **SVRPBench**:

| # | Paradigma | Solver(es) | Notebook |
|---|-----------|-----------|----------|
| 1 | **Métodos Exactos (Branch & Cut)** | `exact-bc` (Gurobi) | `01_exactos_branch_cut.ipynb` |
| 2 | **Metaheurísticas (ACO / Tabu)** | `aco`, `tabu` (oficiales de SVRPBench) | `02_metaheuristicas_aco_tabu.ipynb` |
| 3 | **NCO supervisado** | `nco-sl`, `nco-sl-feas` (Attention Model) | `03_nco_supervisado_attention.ipynb` |
| 4 | **NCO por RL (POMO + AM)** | `nco-rl` (Attention Model + POMO) | `04_nco_rl_pomo.ipynb` |
| 5 | **EHBG-FACS (propuesta)** | `ehbg-facs`, `ehbg-facs-enn` | `05_ehbg_facs.ipynb` |

El notebook `00_setup_y_datos.ipynb` prepara el entorno y el **banco canónico** de
instancias; `06_comparacion_y_estadistica.ipynb` reúne todos los resultados y aplica
**ANOVA / Friedman / Wilcoxon**.

## Cómo garantiza el piso parejo

1. **Mismas instancias.** El banco se genera **una vez** con semillas fijas (notebook 00)
   reutilizando las primitivas oficiales de SVRPBench (`city.City.batch_sample`,
   `time_windows_generator.sample_time_window`) y se cachea en `data/instances/`. Los
   cinco paradigmas cargan exactamente el mismo banco.
2. **Mismos escenarios ξ.** El evaluador estocástico compartido usa **Common Random
   Numbers**: el escenario de la realización *r* se siembra con la semilla de la
   instancia, así dos métodos cualesquiera ven el **mismo** ruido (varianza reducida,
   pruebas estadísticas válidas).
3. **Misma re-puntuación.** Pase lo que pase dentro de cada solver, sus rutas se vuelven
   a puntuar con `svrplab.stochastic.score_routes` bajo el **mismo protocolo**
   (`svrplab.protocol.DEFAULT`): R realizaciones, recurso de 2ª etapa `Q`, CVaR_α, costo
   de flota uniforme.
4. **Mismas métricas y estadística.** `svrplab.metrics` define el esquema canónico y las
   pruebas (supuestos de Shapiro/Levene → ANOVA; si no se cumplen → Friedman + post-hoc
   Wilcoxon con corrección de Holm), con diseño de **bloques por instancia**.

## Uso en Google Colab (Pro / Pro+ recomendado)

1. **Sube tu repo de tesis** a GitHub (o cópialo a Google Drive en
   `MyDrive/TESIS-ANT/`). El pipeline vive en `experiments/colab/`.
2. Abre `notebooks/00_setup_y_datos.ipynb` en Colab y, en *Entorno de ejecución →
   Cambiar tipo de entorno*, elige **GPU** (T4/L4/A100 según tu plan).
3. En la celda de setup, edita `REPO_URL` con la URL de tu repo (o deja `USE_DRIVE=True`
   si lo copiaste a Drive). Ejecuta el notebook 00 para construir el banco.
4. Ejecuta `01`…`05` (en cualquier orden) y luego `06` para la comparación.
   - Mantén **`USE_DRIVE=True`** y los **mismos `SIZES` y `N_INSTANCES`** en todos los
     notebooks: así comparten banco y los resultados se acumulan en la misma raíz.
   - Los modelos neuronales (3/4/5) entrenan una vez y se **cachean** en Drive
     (`data/models/`); las corridas siguientes solo hacen inferencia.

### Aprovechamiento de la GPU
- Paradigmas 3, 4 y 5 usan un **Attention Model tipo Transformer** entrenado en GPU
  (POMO con *mixed precision*; GFlowNet HBG con rollouts vectorizados). El Attention
  Model es la arquitectura del estado del arte (Kool/Kwon) y el "Graph Transformer" de
  la Fase 2 del anteproyecto.
- El evaluador CRN vectoriza la simulación a través de las realizaciones (idéntico al
  bucle de referencia, verificado bit-a-bit).

## Estructura

```
experiments/colab/
  svrplab/                       # paquete compartido por los 5 notebooks
    bootstrap.py                 # clona repo oficial, detecta GPU, rutas (Drive), semillas
    protocol.py                  # condiciones homologadas (fuente única de verdad)
    data.py                      # banco canónico de instancias (primitivas oficiales)
    stochastic.py                # evaluador CRN + recurso de 2ª etapa + CVaR (vectorizado)
    metrics.py                   # esquema de métricas + ANOVA/Friedman/Wilcoxon
    runner.py                    # orquestador (re-puntuación unificada + persistencia)
    viz.py                       # figuras (ruta, convergencia, comparación, tradeoff)
    models/
      transformer.py             # Attention Model (P_F, P_B, F_θ, η, epinet) — GPU
      rollout.py                 # rollouts POMO + decodificación voraz
    solvers/
      exact_bc.py                # 1 — Branch & Cut (Gurobi)
      metaheuristic.py           # 2 — ACO / Tabu oficiales, re-puntuados con CRN
      nco_sl.py                  # 3 — Attention Model supervisado (imita maestro)
      nco_rl.py                  # 4 — Attention Model + POMO (RL)
      ehbg_facs.py               # 5 — HBG-GFlowNet + GFACS + CVaR (+ ENN)
  notebooks/                     # 00..06 (.ipynb)
  scripts/
    verify_evaluator.py          # verifica equivalencia del evaluador y la generación
    smoke_test.py                # corre los 5 paradigmas extremo a extremo (configs mini)
    build_notebooks.py           # regenera los notebooks
  requirements.txt
```

## EHBG-FACS — mecanismos implementados (Paradigma 5)

Fiel en mecanismo al anteproyecto (implementación propia, no el código oficial de los
papers HBG/GFACS/ENN; las simplificaciones se documentan en `solvers/ehbg_facs.py`):

- **GFlowNet de Balance Híbrido (HBG):** el Attention Model parametriza la política de
  avance `P_F`, la de retroceso `P_B` y el flujo de estado `F_θ(s)`, más `log Z`. Se
  entrena con el objetivo híbrido **L = (1−λ)·TB + λ·DB** (ecuaciones 1 y 2 del
  anteproyecto), con `λ_DB` ponderando Balance de Trayectoria vs. Balance Detallado.
- **Recompensa sensible al riesgo:** `R(x) ∝ exp(−CVaR_α(c+Q)/T)`, estimada por
  simulación Monte Carlo CRN; privilegia rutas robustas ante la cola (retrasos
  log-normales + accidentes de Poisson).
- **GFACS:** la matriz heurística a priori `η` de la GFlowNet siembra una **colonia de
  hormigas**; las hormigas muestrean `∝ τ_ACO^α · η^β`, las trayectorias más robustas
  (menor CVaR) actualizan la feromona, y las soluciones exitosas alimentan un **búfer de
  repetición fuera de política** que retroalimenta el entrenamiento de la GFlowNet.
- **ENN (Fase 5, opcional):** cabeza *epinet* indexada para cuantificar la incertidumbre
  epistémica y guiar la exploración (`ehbg-facs-enn`).

## Notas de escalabilidad

- **Gurobi (P1):** la licencia restringida (`pip install gurobipy`) cubre n≤50 con la
  formulación no dirigida. Para n>~63 (escala media del anteproyecto) usa la **licencia
  académica gratuita** de Gurobi.
- **`SIZES`:** por defecto `[10, 20, 50]` (comparación completa con el exacto en licencia
  libre). Para la escala media del anteproyecto (50–300), extiende `SIZES` y usa licencia
  académica para el exacto; los paradigmas neuronales escalan sin cambios.
- **`N_INSTANCES`:** 30 por defecto (rigor para ANOVA/Wilcoxon). Para una primera corrida
  rápida, baja a 5.

## Verificación local (sin GPU)

```bash
# Equivalencia del evaluador CRN y generación de instancias:
PYTHONPATH=experiments/colab python experiments/colab/scripts/verify_evaluator.py \
    --official experiments/svrp/third_party/svrpbench

# Los 5 paradigmas extremo a extremo (configs minúsculas, CPU):
PYTHONPATH=experiments/colab python experiments/colab/scripts/smoke_test.py \
    --official experiments/svrp/third_party/svrpbench
```

## Relación con `experiments/svrp/`

Este pipeline está **basado** en las implementaciones preliminares de `experiments/svrp`
(evaluador CRN, solvers exacto y metaheurístico, esquema NCO), pero: (i) reorganizado en
un paquete autocontenido para Colab; (ii) con los paradigmas 3/4 migrados del Pointer
Network LSTM al **Attention Model Transformer** en GPU; (iii) añadiendo la propuesta
**EHBG-FACS**; y (iv) con banco canónico compartido y validación estadística integrada.
