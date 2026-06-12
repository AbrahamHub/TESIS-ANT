"""Implementación 4/5 — NCO determinista por RL (POMO / Attention Model).

Representa el paradigma "NCO (RL determinista) (POMO, AM)" del Cuadro 1: una política
neuronal entrenada con **aprendizaje por refuerzo** (REINFORCE), sin etiquetas, usando
el **costo de la ruta como recompensa**. Implementa la estrategia **POMO** (Policy
Optimization with Multiple Optima): por cada instancia se generan N trayectorias que
**arrancan en nodos distintos**, y se usa la **media de sus costos como línea base
compartida** (bajo sesgo y baja varianza) — el truco que estabiliza POMO [4].

Diseño (auto-contenido, sin RL4CO): reutiliza la arquitectura del paradigma 3
(``nco_sl.PointerNet``: codificador LSTM + atención puntero con máscara de factibilidad),
pero cambia la **señal de entrenamiento**: en lugar de imitar etiquetas (entropía
cruzada), maximiza ``E[-costo]`` por REINFORCE.

  * **Recompensa = costo determinista** (tiempo de viaje nominal ``τ``), SIN estocasticidad
    en el entrenamiento → es la "NCO determinista". Por eso, evaluada bajo ξ (CRN), exhibe
    la **fragilidad ante la estocasticidad** que señala el anteproyecto.
  * **Inferencia**: decodificación voraz multi-start (POMO) → mejor ruta; rápida (ms).
  * **Puntuación**: con el evaluador estocástico compartido (CRN), comparable con el resto.

Fidelidad y limitaciones (revisión — declaración honesta):
  * **Q1 — arquitectura:** se aplica el *esquema de entrenamiento* POMO (multi-start +
    línea base compartida) sobre una **Pointer Network (LSTM)**, NO sobre el *Attention
    Model* (codificador transformer multi-cabeza) de Kwon/Kool. El esquema POMO es fiel;
    la arquitectura es la del linaje Pointer Network, no la del AM.
  * **Q2 — potencia (resuelto tras Q4/Q5):** con el log-prob del nodo forzado excluido
    (Q4) y el bonus de entropía (Q5), la política es **casi óptima a escala pequeña**
    (gap ~1.6 % en n=10, ~2.6 % en n=20) y **supera a la NCO supervisada** (paradigma 3) —
    coherente con la literatura (RL > supervisado). El gap previo (~50 %) era el bug de Q4,
    no falta de potencia. *Quedan limitaciones declaradas:* arquitectura LSTM (no AM, Q1),
    sin probar a gran escala ni con *instance augmentation*; para la comparación final
    conviene validar en n grande o usar el AM/RL4CO oficial.
  * **Q3 — origen:** SVRPBench no envuelve un POMO (el suyo vive en RL4CO); esta es una
    implementación propia. La fidelidad al benchmark se limita a instancias + evaluador CRN.
  * **Q6 — métrica vs objetivo:** optimiza ``τ`` (tiempo nominal); la comparación usa
    ``E[c]`` (CRN). Coinciden de cerca aquí, pero ``exact-bc`` es óptimo para ``τ``, no
    necesariamente para ``E[c]``.

``import torch`` diferido (paradigmas 1–2 no lo requieren).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from .._bootstrap import Instance, Solution, Solver, register_solver, SVRP_ROOT
from .. import stochastic, io as svrp_io
from .nco_sl import _build_model_cls, node_features, _feasible_mask

_MODEL_DIR = SVRP_ROOT / "data" / "models"


# --------------------------------------------------------------------------- #
# Rollout POMO (vectorizado sobre B instancias × N nodos de inicio)
# --------------------------------------------------------------------------- #


def _rollout_cost(model, feat, demand, cap, tau, device, *, sample=True):
    """Genera N = (n−1) trayectorias por instancia (una por nodo de inicio, POMO) y
    devuelve ``(logp, cost, entropy)`` por trayectoria, vectorizado. No construye rutas
    (solo entrenamiento). ``tau`` (B,n,n) = tiempo de viaje nominal.

    Q4: el primer nodo está **forzado** (POMO), así que su log-prob NO se acumula en
    ``logp`` (no es una decisión de la política). Q5: se acumula la entropía por paso
    para regularizar (mitiga el colapso de modo)."""
    import torch
    B, n, _ = feat.shape
    N = n - 1                                    # un inicio por cliente
    depot = 0
    idxB = torch.arange(B, device=device).repeat_interleave(N)   # (BN,) -> instancia
    BN = B * N
    feat_e, demand_e, cap_e = feat[idxB], demand[idxB], cap[idxB]
    emb, enc, (h, c) = model.encode(feat_e)
    visited = torch.zeros(BN, n, dtype=torch.bool, device=device)
    rem = cap_e.clone()
    last = torch.full((BN,), depot, dtype=torch.long, device=device)
    logp = torch.zeros(BN, device=device)
    entropy = torch.zeros(BN, device=device)
    cost = torch.zeros(BN, device=device)
    arange = torch.arange(BN, device=device)
    start = (torch.arange(N, device=device).repeat(B)) + 1        # clientes 1..n-1
    for t in range(3 * n + 2):
        if bool(visited[:, 1:].all()):
            break
        dec_in = torch.cat([emb[arange, last], (rem / cap_e).unsqueeze(1)], dim=1)
        h, c = model.dec_cell(dec_in, (h, c))
        mask = _feasible_mask(visited, demand_e, rem, last, depot)
        scores = model.attention(enc, h, mask)
        logsm = torch.log_softmax(scores, dim=1)
        if t == 0:
            nxt = start                                          # POMO: inicio forzado (Q4: no entra en logp)
        else:
            if sample:
                nxt = torch.multinomial(torch.softmax(scores, dim=1), 1).squeeze(1)
            else:
                nxt = scores.argmax(1)
            logp = logp + logsm[arange, nxt]
            p = torch.softmax(scores, dim=1)
            entropy = entropy - (p * torch.nan_to_num(logsm, neginf=0.0)).sum(1)  # Q5
        cost = cost + tau[idxB, last, nxt]
        is_depot = nxt == depot
        rem = torch.where(is_depot, cap_e, rem - demand_e[arange, nxt])
        cust = ~is_depot
        visited[arange[cust], nxt[cust]] = True
        last = nxt
    cost = cost + tau[idxB, last, depot]                          # cierre al depósito
    return logp.view(B, N), cost.view(B, N), entropy.view(B, N)


def _decode_greedy(model, instance, device="cpu", depot=0):
    """Inferencia: decodificación voraz multi-start (POMO) → mejor ruta por costo nominal."""
    import torch
    feat = torch.tensor(node_features(instance, depot)[None], device=device)
    demand = torch.tensor(np.asarray(instance.demands, np.float32)[None], device=device)
    cap = torch.tensor([float(instance.vehicle_capacities[0])], device=device)
    dist = stochastic.euclidean_int_matrix(instance.locations)
    tau = stochastic.nominal_time_matrix(dist, stochastic.representative_time(instance, depot))
    n = feat.shape[1]
    N = n - 1
    feat_e = feat.expand(N, n, 5).contiguous()
    demand_e = demand.expand(N, n).contiguous()
    cap_e = cap.expand(N).contiguous()
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)
    seqs = [[] for _ in range(N)]
    with torch.no_grad():
        emb, enc, (h, c) = model.encode(feat_e)
        visited = torch.zeros(N, n, dtype=torch.bool, device=device)
        rem = cap_e.clone(); last = torch.full((N,), depot, dtype=torch.long, device=device)
        cost = torch.zeros(N, device=device)
        arange = torch.arange(N, device=device)
        start = torch.arange(N, device=device) + 1
        for t in range(3 * n + 2):
            if bool(visited[:, 1:].all()):
                break
            dec_in = torch.cat([emb[arange, last], (rem / cap_e).unsqueeze(1)], dim=1)
            h, c = model.dec_cell(dec_in, (h, c))
            mask = _feasible_mask(visited, demand_e, rem, last, depot)
            scores = model.attention(enc, h, mask)
            nxt = start if t == 0 else scores.argmax(1)
            cost = cost + tau_t[last, nxt]
            is_depot = nxt == depot
            rem = torch.where(is_depot, cap_e, rem - demand_e[arange, nxt])
            for r in range(N):
                seqs[r].append(int(nxt[r]) if not bool(is_depot[r]) else depot)
            cust = ~is_depot
            visited[arange[cust], nxt[cust]] = True
            last = nxt
        cost = cost + tau_t[last, depot]
    best = int(cost.argmin().item())
    # partir la secuencia de la mejor trayectoria en rutas
    routes, cur = [], []
    for node in seqs[best]:
        if node == depot:
            if cur:
                routes.append(cur); cur = []
        else:
            cur.append(node)
    if cur:
        routes.append(cur)
    return routes


# --------------------------------------------------------------------------- #
# Entrenamiento POMO-REINFORCE
# --------------------------------------------------------------------------- #


def train_pomo(train_sizes=(10, 20), steps_per_size: int = 1000, batch: int = 64,
               hidden: int = 128, dropout: float = 0.0, lr: float = 1e-3,
               entropy_coef: float = 0.02,
               base_seed: int = 88000, device: str = "cpu", verbose: bool = True):
    import torch
    torch.manual_seed(base_seed)
    PointerNet = _build_model_cls()
    model = PointerNet(feat_dim=5, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    rng = np.random.default_rng(base_seed)
    for size in train_sizes:
        for step in range(steps_per_size):
            insts = [svrp_io.generate_instance(size, seed=int(rng.integers(1, 2**31)),
                                               capacity_mode="binding") for _ in range(batch)]
            feat = torch.tensor(np.stack([node_features(i) for i in insts]), device=device)
            demand = torch.tensor(np.stack([np.asarray(i.demands, np.float32) for i in insts]), device=device)
            cap = torch.tensor([float(i.vehicle_capacities[0]) for i in insts], device=device)
            tau = torch.tensor(np.stack([stochastic.nominal_time_matrix(
                stochastic.euclidean_int_matrix(i.locations),
                stochastic.representative_time(i)).astype(np.float32) for i in insts]), device=device)
            logp, cost, entropy = _rollout_cost(model, feat, demand, cap, tau, device, sample=True)
            baseline = cost.mean(1, keepdim=True)                 # POMO: línea base compartida
            adv = (cost - baseline) / (cost.std(1, keepdim=True) + 1e-6)
            # REINFORCE (minimiza costo) − bonus de entropía (Q5: mitiga colapso de modo)
            loss = (adv.detach() * logp).mean() - entropy_coef * entropy.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if verbose and (step + 1) % 150 == 0:
                print(f"[nco-rl] n={size} step {step+1}/{steps_per_size}  costo_medio={float(cost.mean()):.1f}")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@register_solver("nco-rl")
class NCOReinforce(Solver):
    """Política POMO (REINFORCE multi-start) + inferencia rápida + evaluación CRN."""

    def __init__(
        self,
        *,
        train_sizes=(10, 20),
        steps_per_size: int = 1000,
        batch: int = 64,
        hidden: int = 128,
        device: str = "cpu",
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        accident_scale: float = 1.0,
        train_seed: int = 88000,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.train_sizes = tuple(train_sizes)
        self.steps_per_size = steps_per_size
        self.batch = batch
        self.hidden = hidden
        self.device = device
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale
        self.train_seed = train_seed
        self.cache = cache
        self.verbose = verbose
        self._model = None
        self._train_time = 0.0

    def _model_path(self) -> Path:
        sz = "-".join(str(s) for s in self.train_sizes)
        return _MODEL_DIR / f"nco_rl_pomo_n{sz}_s{self.steps_per_size}_b{self.batch}_h{self.hidden}.pt"

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        PointerNet = _build_model_cls()
        path = self._model_path()
        if self.cache and path.exists():
            self._model = PointerNet(feat_dim=5, hidden=self.hidden, dropout=0.0).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            return
        t0 = time.time()
        self._model = train_pomo(
            train_sizes=self.train_sizes, steps_per_size=self.steps_per_size, batch=self.batch,
            hidden=self.hidden, base_seed=self.train_seed, device=self.device, verbose=self.verbose,
        )
        self._train_time = time.time() - t0
        if self.cache:
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        self._ensure_model()
        depot = int(instance.metadata.get("depot_index", 0))
        t0 = time.time()
        routes = _decode_greedy(self._model, instance, device=self.device, depot=depot)
        infer_time = time.time() - t0

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot,
        )
        extras = score.as_extras()
        extras.update({
            "n_routes": len(routes), "realizations": R, "train_time_s": self._train_time,
            "method": "POMO-REINFORCE", "train_sizes": list(self.train_sizes),
        })
        return Solution(
            routes=routes, total_cost=score.expected_cost, runtime=infer_time,
            feasibility=score.feasibility, cvr=score.cvr, waiting_time=score.waiting_time,
            robustness=score.robustness, extras=extras,
        )
