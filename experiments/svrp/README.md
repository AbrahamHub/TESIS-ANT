# Experimentos preliminares — SVRPBench (tesis EHBG-FACS)

Harness reproducible para comparar los **5 paradigmas** de optimización del
anteproyecto sobre el Problema de Enrutamiento de Vehículos Estocástico (SVRP),
usando instancias y el modelo estocástico del benchmark **SVRPBench**.

Cada paradigma se implementa como un *solver* que produce rutas; **todas** las
rutas se puntúan con el mismo evaluador estocástico (`svrpx/stochastic.py`) para
garantizar comparabilidad.

| # | Paradigma | Estado | Solver | Salidas |
|---|-----------|--------|--------|---------|
| 1 | **Métodos Exactos (Branch & Cut)** | ✅ implementado | `exact-bc` (CVRP) · `exact-bc-tw` (CVRPTW) | `*/01_exact/` |
| 2 | **Metaheurísticas (ACO / Tabu)** | ✅ implementado | `aco` · `tabu` (oficiales, re-puntuados con CRN) | `*/02_metaheuristic/` |
| 3 | **NCO (supervisado)** | ✅ implementado | `nco-sl` (Pointer Network, imita a exact-bc) | `*/03_nco_supervised/` |
| 4 | NCO determinista (POMO / AM) | ⏳ pendiente | — | `*/04_nco_pomo_am/` |
| 5 | **EHBG-FACS** (propuesta) | ⏳ pendiente | — | `*/05_ehbg_facs/` |

Las salidas se organizan por paradigma: `results/<NN_slug>/` y `figures/<NN_slug>/`
(p. ej. `01_exact`, `02_metaheuristic`). Un run que mezcle paradigmas escribe en
`*/cross/`. El mapa solver→carpeta vive en `svrpx/paradigms.py`.

## Estructura

```
experiments/svrp/
  src/svrpx/
    _bootstrap.py        # pone el paquete oficial vrp_bench en el sys.path
    stochastic.py        # evaluador estocástico compartido (CRN + recurso + CVaR)
    metrics.py           # agregación y tablas de métricas
    io.py                # generación/carga de instancias TWCVRP (depósito único)
    paradigms.py         # mapa solver -> paradigma/carpeta de salida
    viz.py               # figuras (ruta, convergencia, histograma+CVaR, barras)
    run_experiment.py    # runner CLI (escribe a results/<NN_slug>/ y figures/<NN_slug>/)
    solvers/
      exact_bc.py        # 1/5 Branch & Cut CVRP (Gurobi)
      exact_bc_tw.py     # 1/5 Branch & Cut CVRPTW (Gurobi, MTZ + ventanas soft)
      metaheuristic.py   # 2/5 ACO y Tabu (oficiales) re-puntuados con CRN
      nco_sl.py          # 3/5 Pointer Network supervisada (PyTorch), imita a exact-bc
  third_party/svrpbench/ # repo oficial clonado (no versionado)
  data/                  # instancias .npz + modelos NCO cacheados (no versionado)
  results/
    01_exact/  02_metaheuristic/  ...  cross/    # CSV + JSON por paradigma (no versionado)
  figures/
    01_exact/  02_metaheuristic/  ...  cross/    # PNG por paradigma (versionado)
```

## Instalación (Apple Silicon / Python 3.13)

```bash
cd experiments/svrp
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install numpy scipy matplotlib pandas gurobipy scikit-learn pillow tqdm
.venv/bin/python -m pip install torch          # solo para el paradigma 3 (NCO supervisado)
# Paquete oficial SVRPBench (se usa vía PYTHONPATH, no se instala con pip):
git clone https://github.com/yehias21/svrpbench third_party/svrpbench
```

## Ejecutar

```bash
# Paradigma 1 — Métodos Exactos (CVRP que ignora ventanas vs CVRPTW que las respeta):
PYTHONPATH=src .venv/bin/python -m svrpx.run_experiment \
    --solver exact-bc,exact-bc-tw --sizes 10,20,50 --instances 3 --realizations 200 --time-limit 90

# Paradigma 2 — Metaheurísticas (ACO vs Tabu):
PYTHONPATH=src .venv/bin/python -m svrpx.run_experiment \
    --solver aco,tabu --sizes 10,20,50 --instances 3 --realizations 200

# Paradigma 3 — NCO supervisado (Pointer Network; entrena/cachea en el 1.er uso):
PYTHONPATH=src .venv/bin/python -m svrpx.run_experiment \
    --solver nco-sl --sizes 10,20,50 --instances 3 --realizations 200

# Comparación entre paradigmas (escribe en cross/):
PYTHONPATH=src .venv/bin/python -m svrpx.run_experiment \
    --solver exact-bc,aco,tabu,nco-sl --sizes 10,20 --instances 3 --realizations 200
```

El paradigma 3 requiere **PyTorch** (`pip install torch`). `nco-sl` entrena una Pointer
Network imitando las rutas óptimas de `exact-bc` sobre instancias de n=20 (etiquetas
caras), la cachea en `data/models/`, y luego hace **inferencia en milisegundos**. Su
calidad es casi óptima en el tamaño de entrenamiento y se degrada fuera de distribución
(n=50) — la limitación clásica del NCO supervisado que motivó el paso al RL (paradigma 4).

Cada run genera `results/<NN_slug>/comparison_metrics.csv` + `.json`, y en
`figures/<NN_slug>/` por tamaño y solver: `*_routes.png`, `*_costhist.png`,
`*_convergence.png` (solo exactos), más `comparison_metrics.png`.

`exact-bc-tw` usa una formulación dirigida (~n² binarias); n=50 requiere licencia
académica de Gurobi y el runner lo **salta** automáticamente si excede la licencia.

## Licencia de Gurobi

`gurobipy` trae una licencia **restringida** (≤2000 variables / restricciones).
Con la formulación **no dirigida** del solver, los tres tamaños (10/20/50) caben
en esa licencia: n=50 ⇒ 1275 aristas. Para instancias mayores (>~63 clientes) se
necesita la **licencia académica gratuita** de Gurobi:

1. Crear cuenta académica en <https://www.gurobi.com/academia/academic-program-and-licenses/>.
2. Obtener una *Named-User Academic License* y ejecutar `grbgetkey <clave>`
   (deja `~/gurobi.lic`).

## Notas de fidelidad a SVRPBench

- **Modelo estocástico** (`stochastic.py`): las *primitivas* (congestión por mezcla
  gaussiana con picos 8:00/17:00, factor log-normal, accidentes de Poisson con pico a
  las 21:00) reproducen exactamente `travel_time_generator.py`. La *simulación de ruta*
  comparte la **semántica de costo** de `vrp_base._simulate_route_execution`
  (`current_time` crudo, espera por llegada temprana, el costo acumula solo tiempo de
  viaje) y la **semántica de violación** de `_check_feasibility` (ventana en hora-del-día,
  `% 1440`). Se evita la cadena de imports pesada `city → scikit-learn/PIL`.
- **Common Random Numbers (CRN)**: cada realización pre-muestrea un escenario ξ (ruido
  por arco×bucket horario) *independiente de la ruta* y determinista en `(seed, r)`, de
  modo que los 5 paradigmas se evalúan sobre **escenarios idénticos** (comparación
  estadística válida). Nota: SVRPBench muestrea costo y factibilidad por separado; aquí
  ambos usan el mismo ξ (más sólido), así que `expected_cost` comparte la *semántica* del
  costo oficial pero no es bit-a-bit idéntico.
- **Dos baselines exactos**: `exact-bc` (CVRP) **ignora** las ventanas en el MIP;
  `exact-bc-tw` (CVRPTW con MTZ y ventanas soft) **sí** las respeta nominalmente.
  Comparar ambos **aísla** cuánta infactibilidad proviene de la estocasticidad (ξ) y
  cuánta de ignorar las ventanas.
- **Objetivo = tiempo de viaje nominal** (no solo distancia): ambos MIP optimizan
  `τ_ij = d_ij + retraso_de_congestión_determinista(t*)`, y el MTZ propaga el horario con
  `τ`. Así el plan incorpora la congestión *conocida* y la infactibilidad residual bajo ξ
  es atribuible a la incertidumbre, no a un horario subestimado. `det_cost` es entonces el
  tiempo de viaje nominal (comparable con `E[c]`).
- **Branch & Cut**: los cortes RCI/DFJ se separan en soluciones **enteras** (`MIPSOL`,
  exacto) y **fraccionarias** (`MIPNODE`, heurística de componentes) para fortalecer la cota.

## Perillas y salvedades

- `--capacity-mode {binding|official}`: `binding` (default) hace un CVRP genuino donde los
  cortes de capacidad sí actúan, pero **no** es comparable 1:1 con SVRPBench; `official`
  replica `cap = Σ demandas` (no restrictiva), comparable pero capacidad inactiva.
- `--accident-scale s`: multiplica la tasa de accidentes Poisson. Con `s=1` (oficial) los
  accidentes son rarísimos (≈1e-4) y por eso **CVaR ≈ media**; subir `s` (p. ej. 20-50)
  engruesa la cola para que el CVaR/robustez discriminen. La factibilidad estricta tiende a
  0 a partir de n≈20 por la **escala** del benchmark (velocidad=1, distancias 0–1000 vs día
  de 1440 min); el eje informativo es el **costo esperado/CVaR**, no la factibilidad.
- **Pesos de recurso**: `late_penalty` (recurso Q, por minuto de retraso) y `tw_penalty`
  (tardanza nominal en el MIP) valen 1.0 por defecto; `E[c+Q]` y CVaR escalan linealmente
  con `late_penalty` (hiperparámetro de modelado, reportar sensibilidad si se varía).
- **Instancias** (`io.py`): se reutilizan las primitivas oficiales
  `city.City.batch_sample` (ubicaciones) y `time_windows_generator.sample_time_window`
  (ventanas residencial/comercial). Para `num_cities = 1` (todos nuestros tamaños)
  se omite el KMeans sobre la rejilla de 10⁶ puntos (mismo resultado, ~23 s → ~0.01 s).
- **Capacidad restrictiva**: el generador oficial fija `cap = Σ demandas` (no
  restrictiva). Aquí se normaliza a `cap = máx(demanda_máx, ⌈demanda_total/K⌉)` para
  obtener un CVRP genuino donde las desigualdades de capacidad redondeada del
  Branch & Cut sí se separan.
- **Ventanas de tiempo y retrasos = recurso de 2ª etapa**: el MIP exacto optimiza
  el costo determinista (distancia nominal); las ventanas y los retrasos
  estocásticos se evalúan como recurso `Q(ruta, ξ)` (penalización por minutos de
  retraso), y el riesgo de cola se mide con CVaR — alineado con la formulación de
  programación estocástica en dos etapas del anteproyecto.
