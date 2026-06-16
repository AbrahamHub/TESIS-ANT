"""svrplab — pipeline de experimentación EHBG-FACS sobre SVRPBench (Google Colab/GPU).

Paquete compartido por los cinco notebooks de paradigma. Garantiza:
  * el **mismo banco** de instancias (``data``),
  * el **mismo evaluador** estocástico CRN (``stochastic``),
  * el **mismo protocolo** de condiciones homologadas (``protocol``),
  * la **misma agregación/estadística** y figuras (``metrics``, ``viz``).

Flujo típico en un notebook:

    from svrplab import bootstrap, data, protocol, runner
    env = bootstrap.init()                       # GPU + repo oficial + rutas
    bank = data.load_bank(env.paths.instances, sizes=[10,20,50],
                          n_instances=protocol.DEFAULT.instances_per_size)
    df = runner.run_solver("aco", bank, env, protocol.DEFAULT)
"""
from . import bootstrap, protocol, stochastic, data  # noqa: F401

__all__ = ["bootstrap", "protocol", "stochastic", "data"]
