"""Solvers del harness. Importar un módulo registra su solver en el registro
oficial de ``vrp_bench`` mediante ``@register_solver``.

Implementación 1/5: ``exact_bc`` (Métodos Exactos — Branch & Cut, Gurobi).
"""
from . import exact_bc        # noqa: F401  (registra "exact-bc")
from . import exact_bc_tw     # noqa: F401  (registra "exact-bc-tw")
from . import metaheuristic   # noqa: F401  (registra "aco", "tabu")

__all__ = ["exact_bc", "exact_bc_tw", "metaheuristic"]

# NCO supervisado (3/5): requiere PyTorch. Import guardado para no romper los
# paradigmas 1–2 si torch no está instalado.
try:
    from . import nco_sl      # noqa: F401  (registra "nco-sl", "nco-sl-feas")
    from . import nco_rl      # noqa: F401  (registra "nco-rl" — POMO/AM)
    __all__ += ["nco_sl", "nco_rl"]
except ImportError as _e:  # pragma: no cover
    print(f"NCO no disponible (¿falta torch?): {_e}")
