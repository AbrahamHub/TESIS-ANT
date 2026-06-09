"""svrpx — harness de experimentos preliminares para SVRPBench.

Paquete de la tesis EHBG-FACS. Compara 5 paradigmas de optimización para el
Problema de Enrutamiento de Vehículos Estocástico (SVRP) sobre instancias
generadas con el paquete oficial ``vrp_bench`` (SVRPBench), puntuando todas las
soluciones con un evaluador estocástico compartido (``stochastic.py``) para
garantizar comparabilidad.

Implementación 1/5: Métodos Exactos (Branch & Cut) — ver ``solvers/exact_bc.py``.
"""

__all__ = ["stochastic", "metrics", "io", "viz"]
