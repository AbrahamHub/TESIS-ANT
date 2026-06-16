"""Paradigma 4 — NCO determinista por RL (Attention Model + POMO).

Representa el paradigma "NCO (RL determinista) (POMO, AM)" del Cuadro 1. A diferencia
del preliminar (Pointer Network LSTM), aquí se entrena el **Attention Model tipo
Transformer** (Kool/Kwon) con la estrategia **POMO** sobre GPU — el SOTA del paradigma.

  * **POMO**: por instancia se generan N = (n−1) trayectorias que **arrancan en nodos
    distintos**, y se usa la **media de sus costos como línea base compartida** (bajo
    sesgo y baja varianza) que estabiliza el entrenamiento.
  * **Recompensa = costo determinista** (tiempo de viaje nominal τ), SIN estocasticidad
    en el entrenamiento → es la "NCO determinista". Por eso, evaluada bajo ξ (CRN),
    exhibe la fragilidad ante la estocasticidad que señala el anteproyecto.
  * **Inferencia**: decodificación voraz multi-start (ms).
  * **Puntuación**: con el evaluador estocástico compartido (CRN).

Se añade un bonus de entropía para mitigar el colapso de modo (regularización estándar
en NCO-RL).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vrp_bench.core import Instance, Solution, Solver
from .. import stochastic, data as svrp_data
from ..models import transformer as T
from ..models import rollout as R


def train_pomo(train_sizes=(10, 20), steps_per_size=1000, batch=64, embed_dim=128,
               n_heads=8, n_layers=3, lr=1e-4, entropy_coef=0.02, base_seed=88000,
               device="cpu", verbose=True):
    import torch
    torch.manual_seed(base_seed)
    model = T.AttentionModel(embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    rng = np.random.default_rng(base_seed)
    history = []
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for size in train_sizes:
        for step in range(steps_per_size):
            insts = [svrp_data.generate_instance(size, seed=int(rng.integers(1, 2**31)),
                                                 capacity_mode="binding") for _ in range(batch)]
            feat, demand, cap, tau = T.instance_tensors(insts, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logp, cost, entropy = R.pomo_rollout(model, feat, demand, cap, tau, sample=True)
                baseline = cost.mean(1, keepdim=True)                 # línea base compartida POMO
                adv = (cost - baseline) / (cost.std(1, keepdim=True) + 1e-6)
                loss = (adv.detach() * logp).mean() - entropy_coef * entropy.mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            history.append(float(cost.mean()))
            if verbose and (step + 1) % 200 == 0:
                print(f"[nco-rl] n={size} paso {step+1}/{steps_per_size}  costo_medio={float(cost.mean()):.1f}")
    model.eval()
    return model, history


class NCOReinforce(Solver):
    """Política AM entrenada por POMO-REINFORCE + inferencia rápida + evaluación CRN."""
    name = "nco-rl"

    def __init__(self, *, train_sizes=(10, 20), steps_per_size=1000, batch=64,
                 embed_dim=128, n_heads=8, n_layers=3, entropy_coef=0.02, device="cpu",
                 default_realizations=200, alpha=0.95, late_penalty=1.0, accident_scale=1.0,
                 train_seed=88000, models_dir=None, cache=True, verbose=True):
        self.train_sizes = tuple(train_sizes)
        self.steps_per_size = steps_per_size
        self.batch = batch
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.entropy_coef = entropy_coef
        self.device = device
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale
        self.train_seed = train_seed
        self.models_dir = Path(models_dir) if models_dir else Path.cwd()
        self.cache = cache
        self.verbose = verbose
        self._model = None
        self._train_time = 0.0
        self.history = []

    def _model_path(self) -> Path:
        sz = "-".join(str(s) for s in self.train_sizes)
        return self.models_dir / f"nco_rl_pomo_am_n{sz}_s{self.steps_per_size}_b{self.batch}_h{self.embed_dim}.pt"

    def ensure_model(self):
        if self._model is not None:
            return
        import torch
        path = self._model_path()
        if self.cache and path.exists():
            self._model = T.AttentionModel(embed_dim=self.embed_dim, n_heads=self.n_heads,
                                           n_layers=self.n_layers).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            if self.verbose:
                print(f"[nco-rl] modelo cargado de cache: {path.name}")
            return
        t0 = time.time()
        self._model, self.history = train_pomo(
            train_sizes=self.train_sizes, steps_per_size=self.steps_per_size, batch=self.batch,
            embed_dim=self.embed_dim, n_heads=self.n_heads, n_layers=self.n_layers,
            entropy_coef=self.entropy_coef, base_seed=self.train_seed, device=self.device,
            verbose=self.verbose)
        self._train_time = time.time() - t0
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        self.ensure_model()
        depot = int(instance.metadata.get("depot_index", 0))
        feat, demand, cap, tau = T.instance_tensors([instance], self.device)
        t0 = time.time()
        routes = R.greedy_decode(self._model, feat, demand, cap, tau, depot)[0]
        infer_time = time.time() - t0

        Rz = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=Rz, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)
        extras = score.as_extras()
        extras.update({"n_routes": len(routes), "realizations": Rz, "train_time_s": self._train_time,
                       "method": "POMO-REINFORCE", "architecture": "AttentionModel",
                       "train_sizes": list(self.train_sizes)})
        return Solution(routes=routes, total_cost=score.expected_cost, runtime=infer_time,
                        feasibility=score.feasibility, cvr=score.cvr,
                        waiting_time=score.waiting_time, robustness=score.robustness, extras=extras)
