"""Verifica que el backend vectorizado del evaluador CRN coincide con el bucle de
referencia (numéricamente, bit-a-bit dentro de tolerancia flotante) y que la
generación de instancias canónicas funciona. Uso local/CI:

    PYTHONPATH=experiments/colab python experiments/colab/scripts/verify_evaluator.py \
        --official experiments/svrp/third_party/svrpbench
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="ruta al repo svrpbench clonado")
    args = ap.parse_args()

    repo = Path(args.official).resolve()
    for p in (str(repo), str(repo / "vrp_bench")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from svrplab import data, stochastic

    print("== generación de instancias ==")
    for n in (10, 20, 50):
        inst = data.generate_instance(n, seed=12345 + n, capacity_mode="binding")
        assert inst.num_nodes == n + 1
        assert inst.demands[0] == 0.0
        assert inst.time_windows.shape == (n + 1, 2)
        print(f"  n={n}: nodos={inst.num_nodes} veh={inst.num_vehicles} "
              f"cap={inst.vehicle_capacities[0]:.0f} ok")

    print("== equivalencia vectorizado vs bucle (CRN) ==")
    rng = np.random.default_rng(0)
    inst = data.generate_instance(20, seed=777, capacity_mode="binding")
    n = inst.num_nodes
    customers = list(range(1, n))
    rng.shuffle(customers)
    routes = [customers[:7], customers[7:14], customers[14:]]   # 3 rutas arbitrarias

    for R in (1, 13, 64):
        a = stochastic.score_routes(inst, routes, num_realizations=R, seed=777,
                                    vectorized=False)
        b = stochastic.score_routes(inst, routes, num_realizations=R, seed=777,
                                    vectorized=True, chunk=5)
        for fld in ("expected_cost", "expected_total", "cvar", "feasibility",
                    "cvr", "waiting_time", "robustness", "tw_violations"):
            va, vb = getattr(a, fld), getattr(b, fld)
            assert abs(va - vb) < 1e-9, f"R={R} {fld}: {va} != {vb}"
        assert np.allclose(a.cost_samples, b.cost_samples, atol=1e-9)
        assert np.allclose(a.total_samples, b.total_samples, atol=1e-9)
        print(f"  R={R}: E[c]={a.expected_cost:.3f}  E[c+Q]={a.expected_total:.3f}  "
              f"CVaR={a.cvar:.3f}  feas={a.feasibility:.2f}  (idéntico)")

    print("\nOK: evaluador vectorizado ≡ bucle de referencia; datos canónicos correctos.")


if __name__ == "__main__":
    main()
