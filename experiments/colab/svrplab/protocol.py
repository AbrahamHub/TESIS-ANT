"""Protocolo experimental homologado — **fuente única de verdad**.

Para que la comparación de los cinco paradigmas sea justa y replicable ("piso
parejo"), todas las condiciones de evaluación se definen aquí y se importan desde
los cinco notebooks. Cambiar una condición en un solo lugar la cambia para todos.

Pilares (alineados con el informe técnico y el anteproyecto):

  * **Mismas instancias**: misma lista, mismo ``seed`` por instancia (ver ``data.py``).
  * **Mismos escenarios ξ**: Common Random Numbers (CRN) sembrados por la semilla de
    la instancia ⇒ los 5 métodos ven el MISMO ruido en cada realización.
  * **Misma re-puntuación**: pase lo que pase dentro de cada solver, sus rutas se
    vuelven a puntuar con el evaluador estocástico compartido y estos parámetros.
  * **Misma medida de riesgo**: CVaR_α (α = 0.95) sobre el costo total con recurso.
  * **Mismo costo de flota**: ``vehicle_fixed_cost`` uniforme (0 = solo viaje).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Protocol:
    """Condiciones experimentales idénticas para los cinco paradigmas."""

    # --- Evaluación estocástica (CRN + recurso de 2ª etapa + CVaR) -------------
    realizations: int = 200          # nº de escenarios Monte Carlo por instancia
    alpha: float = 0.95              # nivel de CVaR (peor 1-α de los escenarios)
    late_penalty: float = 1.0        # recurso Q: penalización por minuto de retraso
    accident_scale: float = 1.0      # ×1 = fiel a SVRPBench (accidentes rarísimos)
    vehicle_fixed_cost: float = 0.0  # costo fijo por ruta usada (0 = solo viaje)
    n_buckets: int = 24              # cubetas horarias del escenario ξ

    # --- Generación de instancias (depósito único, ventanas heterogéneas) ------
    capacity_mode: str = "binding"   # CVRP genuino (capacidad activa)
    base_seed: int = 12345           # semilla base del banco de instancias
    instances_per_size: int = 30     # ≥30 para ANOVA/Wilcoxon (rigor estadístico)

    # --- Validación estadística ------------------------------------------------
    significance: float = 0.05       # α para ANOVA / pruebas no paramétricas

    def eval_kwargs(self) -> dict:
        """kwargs que consume ``stochastic.score_routes`` (re-puntuación unificada)."""
        return dict(
            num_realizations=self.realizations,
            alpha=self.alpha,
            late_penalty=self.late_penalty,
            accident_scale=self.accident_scale,
            vehicle_fixed_cost=self.vehicle_fixed_cost,
            n_buckets=self.n_buckets,
        )

    def as_dict(self) -> dict:
        return asdict(self)


# Instancia por defecto que importan los notebooks. Para barridos de sensibilidad,
# crear un Protocol(...) con los campos cambiados y pasarlo explícitamente.
DEFAULT = Protocol()

# Mapa solver -> (índice de paradigma, slug de carpeta de salida). Mantiene los
# resultados de cada paradigma separados y reconocibles.
PARADIGM_OF = {
    "exact-bc": (1, "01_exact"),
    "exact-bc-tw": (1, "01_exact"),
    "aco": (2, "02_metaheuristic"),
    "tabu": (2, "02_metaheuristic"),
    "nco-sl": (3, "03_nco_supervised"),
    "nco-sl-feas": (3, "03_nco_supervised"),
    "nco-rl": (4, "04_nco_pomo_am"),
    "ehbg-facs": (5, "05_ehbg_facs"),
    "ehbg-facs-enn": (5, "05_ehbg_facs"),
}


def paradigm_slug(solver_names) -> str:
    """Carpeta de salida para un conjunto de solvers (``cross`` si se mezclan)."""
    slugs = {PARADIGM_OF.get(s, (0, "cross"))[1] for s in solver_names}
    return next(iter(slugs)) if len(slugs) == 1 else "cross"
