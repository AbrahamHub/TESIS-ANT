"""Solvers de los cinco paradigmas. Cada uno expone una clase con
``solve(instance, *, num_realizations) -> vrp_bench.core.Solution``; el runner común
los corre sobre el banco canónico y re-puntúa con el evaluador compartido.

Import diferido: los solvers neuronales (3/4/5) importan torch solo al instanciarse,
y el exacto (1) importa gurobipy solo en ``solve``; así el paquete carga aunque falten
dependencias de un paradigma que no se va a usar.
"""
