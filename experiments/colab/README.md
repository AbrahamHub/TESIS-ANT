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
| 5 | **EHBG-FACS (propuesta)** | `ehbg-facs`, `ehbg-facs-enn`, `facs-dist` | `05_ehbg_facs.ipynb` |

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
5. **Búsqueda y evaluación disjuntas (anti-fuga).** Un solver que puntúe internamente sus
   candidatas —el enjambre GFACS, la selección best-of-K de las metaheurísticas— lo hace
   sobre escenarios ξ **disjuntos** de los de evaluación: busca en `r ∈ [R_eval, R_eval+R_s)`
   y se le mide en `r ∈ [0, R_eval)`. Sin esta separación el solver elige el mínimo sobre la
   misma muestra con la que se le mide, y la ventaja observada mezcla calidad real con
   **sesgo de selección**. Lo fija `protocol.search_offset` y lo audita el notebook 06.
6. **Presupuesto de búsqueda declarado.** Cada solver reporta en `search_budget` cuántas
   candidatas puntuó antes de elegir. Comparar métodos poblacionales sin igualar ese número
   no es comparar los métodos, es comparar presupuestos.

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
- **ENN (Fase 5) — H3 de extremo a extremo:** el epinet lleva una **red a priori congelada**
  (Osband et al.), que es la que genera incertidumbre antes de ver datos; la matriz heurística
  `η` queda **indexada por z**, de modo que la incertidumbre llega a la inferencia y no sólo al
  entrenamiento; y el ACO muestrea `∝ τ^α·η^β·(1+κ·u)`, donde `u` es la dispersión de `η` entre
  índices: el **bono explícito de exploración** que H3 enuncia. Con `kappa_epi=0` el bono se
  apaga y queda exactamente EHBG-FACS base, así que la ablación con/sin ENN es limpia.
- **Ablación de atribución (`facs-dist`):** mismo muestreador, mismo presupuesto de candidatas,
  misma regla de feromona y los mismos escenarios de búsqueda, pero sembrado con el prior
  clásico `η = 1/d` en vez de la matriz aprendida. **Es el control que decide si la red aporta
  algo**: si `ehbg-facs` no supera a `facs-dist`, la ventaja frente a los baselines viene del
  presupuesto de búsqueda, no de la GFlowNet. No entrena; corre en CPU en segundos.

## Notas de escalabilidad

- **Gurobi (P1):** la licencia restringida (`pip install gurobipy`) cubre `exact-bc`
  (no dirigido) hasta n≈62 y `exact-bc-tw` (dirigido MTZ) hasta n≈42. El notebook 01
  resuelve los sub-bancos `n ≤ EXACT_MAX_N` / `n ≤ TW_MAX_N` y **detecta automáticamente**
  una licencia académica WLS si guardas `gurobi.lic` en Drive (`GRB_LICENSE_FILE`).
- **Buscar satisfacer las restricciones + siempre generar ruta:** `exact-bc-tw` penaliza
  la tardanza nominal en el objetivo (CVRPTW soft — las ventanas duras son casi
  insatisfacibles por el caveat de escala del benchmark); ambos exactos arrancan de una
  incumbente golosa NN+capacidad (**MIP start**) y caen a ella (`fallback=True`) si el
  límite de tiempo expira sin incumbente — nunca devuelven vacío. `nco-rl` entrena con
  **recompensa consciente de ventanas** (`tw_penalty`: costo nominal + tardanza nominal;
  determinista, sin ξ) y `nco-sl` elige su multi-start por costo+tardanza en inferencia.
  EHBG-FACS ya optimiza CVaR(c+Q), que internaliza las ventanas vía el recurso Q.
- **`SIZES`:** por defecto `[10, 20, 50, 100, 200, 300]` (escala pequeña y media del
  anteproyecto). El exacto se limita a `EXACT_MAX_N`; los demás paradigmas cubren todo.
- **`N_INSTANCES`:** 5 por defecto (corrida exploratoria rápida). **Con 5 bloques el
  Wilcoxon pareado no puede alcanzar p<0.05** (mínimo bilateral 0.0625): para las
  conclusiones de la tesis usa 30. Las semillas por instancia dependen solo de
  `(base_seed, tamaño, índice)`, así que el banco de 5 es un prefijo exacto del de 30.
- **Paralelismo (`N_JOBS`/`n_jobs`):** las fases CPU (Gurobi, ACO/Tabu, evaluador CRN,
  etiquetas del maestro, enjambre GFACS) se reparten en **procesos fork** (`svrplab.parallel`);
  el GIL impide escalar con hilos y CUDA no es fork-safe, por lo que los solve de GPU quedan
  secuenciales y solo se paraleliza su re-puntuación. En `aco_search` los escenarios ξ se
  **pre-muestrean una vez por instancia** y se comparten entre hormigas (idéntico en
  resultados, mucho más rápido a n grande); cada hormiga usa un RNG propio
  `SeedSequence([seed, iteración, hormiga])`, de modo que el resultado no depende de
  `n_jobs` ni del orden de ejecución. Las K corridas best-of-K de las metaheurísticas se
  puntúan con `score_routes_multi` (muestreo ξ compartido por chunk: idéntico bit a bit,
  costo de muestreo ÷K).
- **Figuras:** toda figura mostrada en los notebooks se exporta también a
  `figures/<paradigma>/` en Drive vía `viz.save_show`.
- **Métricas (alineadas al paper):** el núcleo del CSV reproduce las métricas de
  SVRPBench §4.1 — TC=`expected_cost` (Eq. 15), CVR=`cvr` (Eq. 16), FR=`feasibility`
  (Eq. 17), RT=`runtime`, ROB=`rob_var` (Eq. 18, varianza) y `waiting_time` (Fig. 4) —
  más extensiones declaradas: `E[c+Q]`, `E[Q]`, `CVaR/VaR_α`, `total_std`,
  `worst/best_total`, `n_realizations`, `diversity` (métodos poblacionales) y
  `fallback`. El glosario completo (definición/unidades/dirección/fuente) está en
  `metrics.metrics_glossary()` y se exporta a `results/metrics_glossary.csv`.
  Diferencia declarada de protocolo: el paper promedia 5 realizaciones; aquí 200 con
  CRN (necesarias para estimar CVaR al 95%).
- **Estudio de caso común:** todos los notebooks resuelven y exportan paso a paso la
  MISMA instancia (`CASE_SIZE=20`, `CASE_IDX=0`) con la misma representación
  (`viz.plot_route_progression`); el notebook 06 arma la rejilla método-a-método
  (`*_routes.json`) y superpone las distribuciones de `c+Q` bajo los MISMOS ξ
  (`*_samples.npz`, CRN) — las diferencias entre curvas son atribuibles solo a las rutas.
- **Auditoría de piso parejo:** cada corrida persiste en su `run.json` la huella por
  tamaño del banco (`data.size_fingerprints`); el notebook 06 la verifica con un
  `assert` antes de comparar.
- **Sin interpretación anticipada:** las celdas de cierre de los notebooks 01–05 son
  guías de lectura ("qué observar"), no predicciones; las conclusiones se toman en el
  notebook 06 tras la estadística.

## Robustez de las libretas (pensada para Colab)

Colab se desconecta, la GPU se agota y una instancia ocasional revienta. El pipeline está
construido para que nada de eso cueste horas de cómputo:

- **Reanudación automática.** Cada instancia terminada se anota en un `*_checkpoint.jsonl`.
  Si la sesión se cae, vuelves a ejecutar la MISMA celda y sigue donde iba. El checkpoint
  guarda la huella del banco y del protocolo: si cambias `SIZES`, `N_INSTANCES` o el
  protocolo, **se invalida solo** y recalcula, de modo que nunca se mezclan resultados de
  configuraciones distintas en un mismo CSV.
- **Aislamiento de errores.** Una instancia que falla no tumba la corrida: se anota con
  métricas `NaN` y su mensaje en la columna `error`, y el resto continúa. `load_all_results`
  descarta esas filas antes de la estadística e informa cuántas descartó, así que un fallo
  queda **auditado** en vez de desaparecer del conteo.
- **Presupuesto de tiempo.** `TIME_BUDGET_S` corta limpiamente dejando persistido lo hecho.
- **Verificación previa.** Antes de gastar cómputo, cada notebook comprueba cuatro
  invariantes: CRN determinista en `(seed, r)`, ξ independiente de la ruta, solape
  búsqueda/evaluación **igual a cero**, y equivalencia exacta entre el evaluador vectorizado
  y el bucle de referencia. Si algo falla, **detiene el notebook**: es preferible parar ahí
  que producir resultados inválidos.
- **Validación de la configuración.** `SIZES`, `N_INSTANCES`, `CASE_SIZE`/`CASE_IDX` y la
  disjunción del protocolo se validan con `assert` y mensaje explicativo, y el protocolo
  efectivo se imprime en la salida del notebook para que quede registrado en el `.ipynb`.
- **Comprobación real de persistencia.** Montar Drive no garantiza poder escribir; el setup
  hace una escritura de prueba y avisa si los resultados quedarían sólo en disco efímero.

## Análisis de sensibilidad (`svrplab.sensitivity`)

Dos preguntas que deciden qué se puede afirmar en la tesis, y ninguna exige re-resolver nada:

- `risk_regime_sweep(...)` re-puntúa las MISMAS rutas a varias `accident_scale` y localiza el
  régimen donde el CVaR deja de ser la media. A la tasa oficial la brecha CVaR−E[c+Q] es
  ≈0.01 %: la recompensa sensible al riesgo optimiza algo **numéricamente indistinguible del
  costo medio** y ninguna afirmación sobre robustez de cola es contrastable ahí. Reporta los
  dos regímenes por separado: ×1 como fidelidad estricta al benchmark y la escala de estrés
  como el único régimen donde las hipótesis de riesgo se pueden contrastar.
- `leakage_bias(...)` corre el mismo solver con búsqueda disjunta y con búsqueda fugada, y
  mide la inflación. Sirve para declarar **cuánto valía** el defecto corregido, en vez de sólo
  afirmar que se corrigió. Es la única celda autorizada a usar la configuración sesgada, y su
  salida no alimenta ninguna tabla de resultados.
- `audit_runs(...)` verifica antes de comparar que todos los solvers usaron el mismo protocolo,
  buscaron de forma disjunta y terminaron sin fallos.

## Verificación local (sin GPU)

```bash
# Equivalencia del evaluador CRN y generación de instancias:
PYTHONPATH=experiments/colab python experiments/colab/scripts/verify_evaluator.py \
    --official experiments/svrp/third_party/svrpbench

# Los 5 paradigmas extremo a extremo (configs minúsculas, CPU). Incluye la
# ablación `facs-dist`, la aserción de que NINGÚN solver busca sobre los
# escenarios de evaluación, las pruebas de reanudación y aislamiento de fallos,
# y el barrido de régimen de riesgo:
PYTHONPATH=experiments/colab python experiments/colab/scripts/smoke_test.py \
    --official experiments/svrp/third_party/svrpbench --skip-gurobi
```

## Relación con `experiments/svrp/`

Este pipeline está **basado** en las implementaciones preliminares de `experiments/svrp`
(evaluador CRN, solvers exacto y metaheurístico, esquema NCO), pero: (i) reorganizado en
un paquete autocontenido para Colab; (ii) con los paradigmas 3/4 migrados del Pointer
Network LSTM al **Attention Model Transformer** en GPU; (iii) añadiendo la propuesta
**EHBG-FACS**; y (iv) con banco canónico compartido y validación estadística integrada.
