"""Smoke test (CPU, configs minúsculas) de los cinco paradigmas + runner + métricas.
No mide calidad; solo verifica que todo corre extremo a extremo sin errores.

    PYTHONPATH=experiments/colab python experiments/colab/scripts/smoke_test.py \
        --official experiments/svrp/third_party/svrpbench
"""
import argparse
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--skip-gurobi", action="store_true")
    args = ap.parse_args()

    repo = Path(args.official).resolve()
    for p in (str(repo), str(repo / "vrp_bench")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    from svrplab import data, protocol, metrics, runner
    from svrplab.bootstrap import Env, Paths

    tmp = Path(tempfile.mkdtemp(prefix="svrplab_smoke_"))
    env = Env(paths=Paths(root=tmp, official_repo=repo), device="cpu", official_repo=repo)
    proto = protocol.Protocol(realizations=20, instances_per_size=2)

    print("== banco canónico ==")
    bank = data.build_bank(env.paths.instances, sizes=[10], n_instances=2,
                           base_seed=proto.base_seed, capacity_mode="binding")

    results = {}

    if not args.skip_gurobi:
        print("\n== P1 exact-bc ==")
        from svrplab.solvers.exact_bc import ExactBranchCut
        results["exact-bc"] = runner.run_solver(
            ExactBranchCut(time_limit=20.0), "exact-bc", bank, env, proto, verbose=True)

    print("\n== P2 aco ==")
    from svrplab.solvers.metaheuristic import ACO
    results["aco"] = runner.run_solver(ACO(n_seeds=2), "aco", bank, env, proto, verbose=True)

    print("\n== P3 nco-sl (AM supervisado, mini) ==")
    from svrplab.solvers.nco_sl import NCOSupervised
    sl = NCOSupervised(teacher="aco", train_sizes=(10,), n_per_size=6, epochs=4,
                       embed_dim=32, n_heads=4, n_layers=2, device="cpu",
                       models_dir=env.paths.models, verbose=True)
    results["nco-sl"] = runner.run_solver(sl, "nco-sl", bank, env, proto, verbose=True)

    print("\n== P4 nco-rl (AM+POMO, mini) ==")
    from svrplab.solvers.nco_rl import NCOReinforce
    rl = NCOReinforce(train_sizes=(10,), steps_per_size=10, batch=8, embed_dim=32,
                      n_heads=4, n_layers=2, device="cpu", models_dir=env.paths.models, verbose=True)
    results["nco-rl"] = runner.run_solver(rl, "nco-rl", bank, env, proto, verbose=True)

    print("\n== P5 ehbg-facs (mini) ==")
    from svrplab.solvers.ehbg_facs import EHBGFACS
    e = EHBGFACS(train_sizes=(10,), n_train=8, epochs=2, embed_dim=32, n_heads=4,
                 n_layers=2, batch=4, refine_every=1, aco_train_realizations=10,
                 train_realizations=10, infer_ants=6, infer_iters=3, infer_realizations=15,
                 device="cpu", models_dir=env.paths.models, verbose=True)
    results["ehbg-facs"] = runner.run_solver(e, "ehbg-facs", bank, env, proto, verbose=True)

    print("\n== P5 ehbg-facs-enn (epinet, mini) ==")
    from svrplab.solvers.ehbg_facs import EHBGFACSEpistemic
    een = EHBGFACSEpistemic(train_sizes=(10,), n_train=8, epochs=2, embed_dim=32, n_heads=4,
                            n_layers=2, batch=4, refine_every=1, aco_train_realizations=10,
                            train_realizations=10, infer_ants=6, infer_iters=3,
                            infer_realizations=15, device="cpu", models_dir=env.paths.models,
                            verbose=True)
    results["ehbg-facs-enn"] = runner.run_solver(een, "ehbg-facs-enn", bank, env, proto, verbose=True)

    print("\n== agregación + estadística ==")
    import pandas as pd
    alldf = pd.concat(results.values(), ignore_index=True)
    print(metrics.leaderboard(alldf).to_string(index=False))
    cmp = metrics.compare_solvers(alldf, metric="expected_total", size=10)
    print(metrics.summarize_comparison(cmp))

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
