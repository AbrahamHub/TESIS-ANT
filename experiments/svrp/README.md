# Experimentos preliminares — SVRPBench (tesis EHBG-FACS)

Harness reproducible para comparar los **5 paradigmas** de optimización del
anteproyecto sobre el Problema de Enrutamiento de Vehículos Estocástico (SVRP),
usando instancias y el modelo estocástico del benchmark **SVRPBench**.

Cada paradigma se implementa como un *solver* que produce rutas; **todas** las
rutas se puntúan con el mismo evaluador estocástico (`svrpx/stochastic.py`) para
garantizar comparabilidad.

| # | Paradigma | Estado | Solver |
|---|-----------|--------|--------|
| 1 | **Métodos Exactos (Branch & Cut)** | ✅ implementado | `exact-bc` (Gurobi) |
| 2 | Metaheurísticas (ACO / Tabu) | ⏳ pendiente | — |
| 3 | NCO (RL supervisado) | ⏳ pendiente | — |
| 4 | NCO determinista (POMO / AM) | ⏳ pendiente | — |
| 5 | **EHBG-FACS** (propuesta) | ⏳ pendiente | — |

## Estructura

```
experiments/svrp/
  src/svrpx/
    _bootstrap.py     # pone el paquete oficial vrp_bench en el sys.path
    stochastic.py     # evaluador estocástico compartido (4 vectores + recurso + CVaR)
    metrics.py        # agregación y tablas de métricas
    io.py             # generación/carga de instancias TWCVRP oficiales (depósito único)
    solvers/exact_bc.py  # Branch & Cut (Gurobi)  <-- implementación 1/5
    viz.py            # 4 figuras (ruta, convergencia, histograma+CVaR, barras)
    run_experiment.py # runner CLI
  third_party/svrpbench/   # repo oficial clonado (no versionado)
  data/                    # instancias .npz cacheadas (no versionado)
  results/                 # CSV + JSON de métricas (no versionado)
  figures/                 # PNG para la tesis (versionado)
```

## Instalación (Apple Silicon / Python 3.13)

```bash
cd experiments/svrp
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install numpy scipy matplotlib pandas gurobipy scikit-learn pillow tqdm
# Paquete oficial SVRPBench (se usa vía PYTHONPATH, no se instala con pip):
git clone https://github.com/yehias21/svrpbench third_party/svrpbench
```

## Ejecutar (implementación 1: Branch & Cut)

```bash
PYTHONPATH=src .venv/bin/python -m svrpx.run_experiment \
    --solver exact-bc --sizes 10,20,50 --instances 3 --realizations 200 --time-limit 120
```

Genera `results/exact-bc_metrics.csv`, `results/exact-bc_per_instance.json`, y en
`figures/` por tamaño: `*_routes.png`, `*_convergence.png`, `*_costhist.png`, más
`exact-bc_metrics.png`.

## Licencia de Gurobi

`gurobipy` trae una licencia **restringida** (≤2000 variables / restricciones).
Con la formulación **no dirigida** del solver, los tres tamaños (10/20/50) caben
en esa licencia: n=50 ⇒ 1275 aristas. Para instancias mayores (>~63 clientes) se
necesita la **licencia académica gratuita** de Gurobi:

1. Crear cuenta académica en <https://www.gurobi.com/academia/academic-program-and-licenses/>.
2. Obtener una *Named-User Academic License* y ejecutar `grbgetkey <clave>`
   (deja `~/gurobi.lic`).

## Notas de fidelidad a SVRPBench

- **Modelo estocástico** (`stochastic.py`): réplica textual de
  `travel_time_generator.py` (congestión por mezcla gaussiana con picos 8:00/17:00,
  retardo log-normal, accidentes de Poisson con pico a las 21:00) y de la
  simulación de ruta de `vrp_base.py`. Se evita así la cadena de imports pesada
  `city → scikit-learn/PIL`.
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
