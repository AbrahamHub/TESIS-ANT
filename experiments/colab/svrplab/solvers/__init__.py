"""Solvers de los cinco paradigmas. Cada uno expone una clase con
``solve(instance, *, num_realizations) -> vrp_bench.core.Solution``; el runner común
los corre sobre el banco canónico y re-puntúa con el evaluador compartido.

Import diferido: los solvers neuronales (3/4/5) importan torch solo al instanciarse,
y el exacto (1) importa gurobipy solo en ``solve``; así el paquete carga aunque falten
dependencias de un paradigma que no se va a usar.
"""


# Registro explícito de los solvers de la propuesta y su ablación de atribución.
# Import diferido: `ehbg_facs` importa torch, así que se resuelve al usarse.
__all__ = ["EHBGFACS", "EHBGFACSEpistemic", "FACSDistancePrior"]


def __getattr__(name):
    if name in __all__:
        from . import ehbg_facs as _m
        return getattr(_m, name)
    raise AttributeError(name)
