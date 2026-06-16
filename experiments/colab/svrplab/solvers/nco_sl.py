"""Paradigma 3 — NCO supervisado (Attention Model entrenado por imitación).

Representa el paradigma "NCO (RL supervisado)" del Cuadro 1: una red neuronal entrenada
para **imitar rutas óptimas etiquetadas** por un solucionador caro, con **inferencia
rápida** después. A diferencia del preliminar (que usaba una Pointer Network LSTM), aquí
se usa el **Attention Model tipo Transformer** (``models.transformer``), entrenado en GPU
— la arquitectura del estado del arte.

Diseño:
  * **Etiquetas (caras):** se generan resolviendo instancias de entrenamiento con un
    **maestro configurable** — ``exact-bc`` (óptimo, ignora ventanas; ``nco-sl``) o
    ``aco`` (factible; variante ``nco-sl-feas``). La (in)factibilidad la define el
    maestro, no el paradigma.
  * **Entrenamiento:** imitación con *teacher forcing* y entropía cruzada enmascarada
    sobre la secuencia de la ruta del maestro (clientes con retornos al depósito),
    batched por tamaño en GPU; se canonicaliza el orden de rutas (estable).
  * **Inferencia:** decodificación voraz multi-start del AM (ms); el costo de
    entrenamiento se reporta aparte (amortizado).
  * **Puntuación:** con el evaluador estocástico compartido (CRN).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from vrp_bench.core import Instance, Solution, Solver
from .. import stochastic, data as svrp_data
from ..models import transformer as T
from ..models import rollout as R


# --------------------------------------------------------------------------- #
# Etiquetas del maestro
# --------------------------------------------------------------------------- #


def _make_teacher(name: str):
    if name == "exact-bc":
        from .exact_bc import ExactBranchCut
        return ExactBranchCut(time_limit=30.0, verbose=False)
    if name == "aco":
        from .metaheuristic import ACO
        return ACO(n_seeds=3)
    raise ValueError(f"maestro desconocido: {name}")


def teacher_sequence(routes, locations, depot: int = 0) -> List[int]:
    """Aplana rutas a una secuencia con retornos al depósito, canonicalizada: rutas
    ordenadas por el ángulo de su primer cliente respecto al depósito (estabiliza el
    objetivo ante la multiplicidad de óptimos del aprendizaje supervisado)."""
    locs = np.asarray(locations)
    d = locs[depot]

    def angle(route):
        c = locs[route[0]]
        return np.arctan2(c[1] - d[1], c[0] - d[0])

    seq: List[int] = []
    for r in sorted([list(r) for r in routes if r], key=angle):
        seq.extend(int(c) for c in r)
        seq.append(depot)
    return seq


def _gen_labeled(train_sizes, n_per_size, base_seed, teacher_name, verbose=True):
    teacher = _make_teacher(teacher_name)
    data = []
    k = 0
    for size in train_sizes:
        for _ in range(n_per_size):
            inst = svrp_data.generate_instance(size, seed=base_seed + k, capacity_mode="binding")
            k += 1
            sol = teacher.solve(inst, num_realizations=1)
            if not sol.routes:
                continue
            data.append((inst, T.node_features(inst), teacher_sequence(sol.routes, inst.locations)))
    if verbose:
        print(f"[nco-sl] etiquetas ({teacher_name}): {len(data)} instancias, tamaños {list(train_sizes)}")
    return data


# --------------------------------------------------------------------------- #
# Entrenamiento (teacher forcing, GPU)
# --------------------------------------------------------------------------- #


def _seq_loss(model, feat, demand, cap, target):
    """Entropía cruzada de imitación con teacher forcing sobre el AM."""
    import torch
    import torch.nn.functional as Fnn
    device = feat.device
    B, n, _ = feat.shape
    depot = 0
    emb, graph = model.encode(feat)
    visited = torch.zeros(B, n, dtype=torch.bool, device=device)
    rem = cap.clone()
    last = torch.full((B,), depot, dtype=torch.long, device=device)
    ar = torch.arange(B, device=device)
    loss = torch.zeros((), device=device)
    for t in range(target.shape[1]):
        last_emb = emb[ar, last]
        mask = T.feasible_mask(visited, demand, rem, last, depot)
        tgt = target[:, t]
        valid = tgt != -100
        safe = tgt.clamp(min=0)
        mask[ar[valid], safe[valid]] = True            # el objetivo siempre elegible
        logits, _ = model.step_logits(emb, graph, last_emb, rem / cap, mask)
        if valid.any():
            loss = loss + Fnn.cross_entropy(logits[valid], tgt[valid], reduction="sum")
        nxt = tgt.clamp(min=0)
        is_depot = nxt == depot
        upd = valid
        cust = (~is_depot) & upd
        visited[ar[cust], nxt[cust]] = True
        rem = torch.where(is_depot & upd, cap, rem - torch.where(upd, demand[ar, nxt], torch.zeros_like(rem)))
        last = torch.where(upd, nxt, last)
    return loss / (target != -100).sum().clamp(min=1)


def train_supervised(train_sizes=(10, 20), n_per_size=256, epochs=80, embed_dim=128,
                     n_heads=8, n_layers=3, lr=1e-4, batch_size=64, base_seed=99000,
                     teacher="exact-bc", val_frac=0.1, device="cpu", verbose=True):
    import torch
    from collections import defaultdict
    torch.manual_seed(base_seed)
    model = T.AttentionModel(embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    data = _gen_labeled(train_sizes, n_per_size, base_seed, teacher, verbose)
    by_n = defaultdict(list)
    for d in data:
        by_n[d[0].num_nodes].append(d)

    def tensors(items):
        feats = torch.as_tensor(np.stack([d[1] for d in items]), device=device)
        demands = torch.as_tensor(np.stack([np.asarray(d[0].demands, np.float32) for d in items]), device=device)
        caps = torch.as_tensor([float(d[0].vehicle_capacities[0]) for d in items], device=device)
        max_len = max(len(d[2]) for d in items)
        targets = torch.full((len(items), max_len), -100, dtype=torch.long, device=device)
        for i, d in enumerate(items):
            targets[i, :len(d[2])] = torch.as_tensor(d[2], device=device)
        return feats, demands, caps, targets

    groups = {}
    val_groups = {}
    for nval, items in by_n.items():
        nval_val = max(1, int(len(items) * val_frac))
        val_groups[nval] = tensors(items[:nval_val])
        groups[nval] = tensors(items[nval_val:])

    history = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for nval, (feats, demands, caps, targets) in groups.items():
            Nn = feats.shape[0]
            perm = torch.randperm(Nn, device=device)
            for i in range(0, Nn, batch_size):
                idx = perm[i:i + batch_size]
                loss = _seq_loss(model, feats[idx], demands[idx], caps[idx], targets[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
        history.append(tot / max(1, len(data)))
        if verbose and (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(float(_seq_loss(model, *val_groups[k]).detach()) for k in val_groups) / max(1, len(val_groups))
            print(f"[nco-sl] época {ep+1}/{epochs}  CE_train={history[-1]:.4f}  CE_val={vl:.4f}")
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class NCOSupervised(Solver):
    """Attention Model supervisado (imita a un maestro) + inferencia rápida + CRN."""
    name = "nco-sl"

    def __init__(self, *, teacher: str = "exact-bc", train_sizes=(10, 20), n_per_size=256,
                 epochs=80, embed_dim=128, n_heads=8, n_layers=3, device="cpu",
                 default_realizations=200, alpha=0.95, late_penalty=1.0, accident_scale=1.0,
                 train_seed=99000, models_dir=None, cache=True, verbose=True):
        self.teacher = teacher
        self.train_sizes = tuple(train_sizes)
        self.n_per_size = n_per_size
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
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
        return self.models_dir / (f"nco_sl_{self.teacher}_n{sz}_t{self.n_per_size}"
                                  f"_e{self.epochs}_h{self.embed_dim}.pt")

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
                print(f"[nco-sl] modelo cargado de cache: {path.name}")
            return
        t0 = time.time()
        self._model, self.history = train_supervised(
            train_sizes=self.train_sizes, n_per_size=self.n_per_size, epochs=self.epochs,
            embed_dim=self.embed_dim, n_heads=self.n_heads, n_layers=self.n_layers,
            base_seed=self.train_seed, teacher=self.teacher, device=self.device,
            verbose=self.verbose)
        self._train_time = time.time() - t0
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        import torch
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
        extras.update({"n_routes": len(routes), "realizations": Rz,
                       "train_time_s": self._train_time, "teacher": self.teacher,
                       "train_sizes": list(self.train_sizes), "architecture": "AttentionModel"})
        return Solution(routes=routes, total_cost=score.expected_cost, runtime=infer_time,
                        feasibility=score.feasibility, cvr=score.cvr,
                        waiting_time=score.waiting_time, robustness=score.robustness, extras=extras)


class NCOSupervisedFeasible(NCOSupervised):
    """Variante de control: mismo AM supervisado pero imitando un maestro **factible**
    (``aco``). Separa la limitación del paradigma de la elección de maestro."""
    name = "nco-sl-feas"

    def __init__(self, **kw):
        kw.setdefault("teacher", "aco")
        super().__init__(**kw)
