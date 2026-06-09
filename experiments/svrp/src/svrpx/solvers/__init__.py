"""Solvers del harness. Importar un módulo registra su solver en el registro
oficial de ``vrp_bench`` mediante ``@register_solver``.

Implementación 1/5: ``exact_bc`` (Métodos Exactos — Branch & Cut, Gurobi).
"""
from . import exact_bc      # noqa: F401  (registra "exact-bc")
from . import exact_bc_tw   # noqa: F401  (registra "exact-bc-tw")

__all__ = ["exact_bc", "exact_bc_tw"]
