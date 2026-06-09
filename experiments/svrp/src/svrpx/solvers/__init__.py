"""Solvers del harness. Importar un módulo registra su solver en el registro
oficial de ``vrp_bench`` mediante ``@register_solver``.

Implementación 1/5: ``exact_bc`` (Métodos Exactos — Branch & Cut, Gurobi).
"""
from . import exact_bc  # noqa: F401  (registra "exact-bc")

__all__ = ["exact_bc"]
