"""Implementación 3/5 — NCO con aprendizaje supervisado (Pointer Network).

Representa el paradigma "NCO (RL supervisado)" del Cuadro 1: la línea de **Redes de
Apuntadores (Pointer Networks) recurrentes entrenadas de forma supervisada**
(Vinyals et al.) que **imitan rutas óptimas etiquetadas** por un solucionador caro y
luego hacen **inferencia rápida**.

Diseño (auto-contenido en PyTorch, sin RL4CO):

  * **Etiquetas (caras):** se generan resolviendo instancias de entrenamiento con el
    solver exacto ``exact-bc`` (CVRP óptimo). Esto materializa la "dependencia total de
    datos etiquetados caros" del paradigma.
  * **Modelo:** codificador LSTM sobre los nodos + decodificador autorregresivo con
    **atención tipo puntero** (Bahdanau) que apunta al siguiente nodo, con máscara de
    factibilidad (clientes visitados y capacidad). Tamaño-agnóstico: se entrena en un
    tamaño y se infiere en otros.
  * **Entrenamiento:** imitación con *teacher forcing* y entropía cruzada enmascarada
    sobre la secuencia aplanada de la ruta óptima (clientes con retornos al depósito).
  * **Inferencia:** decodificación voraz con máscaras → rutas; se mide el tiempo de
    **inferencia** (rápido), con el costo de entrenamiento reportado aparte (amortizado).
  * **Puntuación:** con el evaluador estocástico compartido (CRN), comparable con el
    resto de paradigmas. Como imita a ``exact-bc`` (que ignora ventanas), hereda su
    fragilidad ante ventanas bajo ξ — justo el rasgo del paradigma.

Nota: el ``import torch`` se hace de forma diferida para que el paquete cargue aunque
PyTorch no esté instalado (paradigmas 1–2 no lo necesitan).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .._bootstrap import Instance, Solution, Solver, register_solver, SVRP_ROOT
from .. import stochastic, io as svrp_io

_DAY = 1440.0
_MODEL_DIR = SVRP_ROOT / "data" / "models"


# --------------------------------------------------------------------------- #
# Características y etiquetas
# --------------------------------------------------------------------------- #


def node_features(instance: Instance, depot: int = 0) -> np.ndarray:
    """(n, 5): [x/1000, y/1000, demanda/cap, ventana_ini/1440, ventana_fin/1440]."""
    locs = np.asarray(instance.locations, dtype=np.float64)
    n = locs.shape[0]
    dem = np.asarray(instance.demands, dtype=np.float64)
    cap = float(np.asarray(instance.vehicle_capacities, dtype=np.float64).ravel()[0]) or 1.0
    tw = np.asarray(instance.time_windows, dtype=np.float64) if instance.time_windows is not None \
        else np.tile([0.0, _DAY], (n, 1))
    mx = float(np.abs(locs).max()) or 1.0
    feat = np.stack([
        locs[:, 0] / mx, locs[:, 1] / mx, dem / cap,
        tw[:, 0] / _DAY, tw[:, 1] / _DAY,
    ], axis=1)
    return feat.astype(np.float32)


def teacher_sequence(routes: List[List[int]], depot: int = 0) -> List[int]:
    """Aplana rutas (clientes) en la secuencia objetivo del decodificador:
    ``[[1,2],[3]] -> [1, 2, depot, 3, depot]`` (el decodificador arranca en el depósito)."""
    seq: List[int] = []
    for r in routes:
        seq.extend(int(c) for c in r)
        seq.append(depot)
    return seq


# --------------------------------------------------------------------------- #
# Modelo (definido perezosamente para no exigir torch al importar el paquete)
# --------------------------------------------------------------------------- #


def _build_model_cls():
    import torch
    import torch.nn as nn

    class PointerNet(nn.Module):
        def __init__(self, feat_dim: int = 5, hidden: int = 128):
            super().__init__()
            self.hidden = hidden
            self.embed = nn.Linear(feat_dim, hidden)
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.dec_cell = nn.LSTMCell(hidden + 1, hidden)
            # Atención aditiva (Bahdanau): score = v^T tanh(W1 enc + W2 query)
            self.W1 = nn.Linear(hidden, hidden, bias=False)
            self.W2 = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)

        def encode(self, feat):  # feat (B, n, F)
            emb = self.embed(feat)                  # (B, n, H)
            enc, (h, c) = self.encoder(emb)         # enc (B, n, H)
            return emb, enc, (h[-1], c[-1])

        def attention(self, enc, query, mask):      # enc (B,n,H), query (B,H), mask (B,n) bool feasible
            w1 = self.W1(enc)                        # (B,n,H)
            w2 = self.W2(query).unsqueeze(1)         # (B,1,H)
            scores = self.v(torch.tanh(w1 + w2)).squeeze(-1)  # (B,n)
            scores = scores.masked_fill(~mask, float("-inf"))
            return scores

    return PointerNet


# --------------------------------------------------------------------------- #
# Entrenamiento supervisado
# --------------------------------------------------------------------------- #


def _gen_labeled(train_size: int, n_train: int, base_seed: int):
    """Genera instancias de entrenamiento y sus etiquetas óptimas (exact-bc)."""
    from .exact_bc import ExactBranchCut
    teacher = ExactBranchCut(time_limit=20.0, default_realizations=1)
    data = []
    for k in range(n_train):
        inst = svrp_io.generate_instance(train_size, seed=base_seed + k, capacity_mode="binding")
        sol = teacher.solve(inst, num_realizations=1)
        if not sol.routes:
            continue
        feat = node_features(inst)
        seq = teacher_sequence(sol.routes, depot=0)
        data.append((inst, feat, seq))
    return data


def train_pointer_net(train_size: int = 20, n_train: int = 256, epochs: int = 15,
                      hidden: int = 128, lr: float = 1e-3, batch_size: int = 32,
                      base_seed: int = 99000, device: str = "cpu", verbose: bool = True):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(base_seed)
    PointerNet = _build_model_cls()
    model = PointerNet(feat_dim=5, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    data = _gen_labeled(train_size, n_train, base_seed)
    if verbose:
        print(f"[nco-sl] etiquetas generadas: {len(data)} instancias n={train_size}")

    feats = torch.tensor(np.stack([d[1] for d in data]), device=device)  # (N, n, 5)
    demands = torch.tensor(np.stack([np.asarray(d[0].demands, np.float32) for d in data]), device=device)
    caps = torch.tensor([float(d[0].vehicle_capacities[0]) for d in data], device=device)
    seqs = [d[2] for d in data]
    max_len = max(len(s) for s in seqs)
    N, n, _ = feats.shape
    targets = torch.full((N, max_len), -100, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        targets[i, :len(s)] = torch.tensor(s, device=device)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for b0 in range(0, N, batch_size):
            idx = perm[b0:b0 + batch_size]
            fb, db, cb, tb = feats[idx], demands[idx], caps[idx], targets[idx]
            loss = _seq_loss(model, fb, db, cb, tb, device, F)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        if verbose:
            print(f"[nco-sl] epoch {ep+1}/{epochs}  loss={total/N:.4f}")
    model.eval()
    return model


def _seq_loss(model, feat, demand, cap, target, device, F):
    """Pérdida de imitación (teacher forcing) sobre la secuencia objetivo."""
    import torch
    B, n, _ = feat.shape
    emb, enc, (h, c) = model.encode(feat)
    depot = 0
    visited = torch.zeros(B, n, dtype=torch.bool, device=device)
    rem = cap.clone()
    last = torch.full((B,), depot, dtype=torch.long, device=device)
    arange = torch.arange(B, device=device)
    loss = torch.zeros((), device=device)
    steps = target.shape[1]
    for t in range(steps):
        last_emb = emb[arange, last]                      # (B,H)
        dec_in = torch.cat([last_emb, (rem / cap).unsqueeze(1)], dim=1)
        h, c = model.dec_cell(dec_in, (h, c))
        mask = _feasible_mask(visited, demand, rem, last, depot)
        tgt = target[:, t]
        valid = tgt != -100
        # Garantizar que el objetivo del maestro sea siempre elegible: evita logits
        # -inf en el objetivo (pérdida infinita) sin alterar el gradiente del modelo.
        safe = tgt.clamp(min=0)
        mask[arange[valid], safe[valid]] = True
        scores = model.attention(enc, h, mask)            # (B,n)
        if valid.any():
            loss = loss + F.cross_entropy(scores[valid], tgt[valid], reduction="sum")
        # teacher forcing: avanzar con el objetivo (clamp para pasos padded)
        nxt = tgt.clamp(min=0)
        is_depot = nxt == depot
        # actualizar capacidad/visitados solo en pasos válidos
        upd = valid
        cust = (~is_depot) & upd
        visited[arange[cust], nxt[cust]] = True
        rem = torch.where(is_depot & upd, cap, rem - torch.where(upd, demand[arange, nxt], torch.zeros_like(rem)))
        last = torch.where(upd, nxt, last)
    denom = (target != -100).sum().clamp(min=1)
    return loss / denom


def _feasible_mask(visited, demand, rem, last, depot):
    """(B,n) bool: nodos elegibles. Clientes no visitados que caben en capacidad;
    el depósito solo si NO venimos del depósito (evita depósito->depósito)."""
    import torch
    B, n = visited.shape
    arange = torch.arange(B, device=visited.device)
    fits = demand <= (rem.unsqueeze(1) + 1e-6)
    mask = (~visited) & fits
    mask[:, depot] = (last != depot)  # depósito permitido salvo si ya estamos en él
    # si no hay ningún cliente factible, forzar depósito como única opción
    none_feasible = mask[:, 1:].any(dim=1) == False
    mask[none_feasible, depot] = True
    return mask


# --------------------------------------------------------------------------- #
# Inferencia voraz
# --------------------------------------------------------------------------- #


def greedy_routes(model, instance: Instance, device: str = "cpu", depot: int = 0) -> List[List[int]]:
    import torch
    feat = torch.tensor(node_features(instance, depot)[None], device=device)  # (1,n,5)
    demand = torch.tensor(np.asarray(instance.demands, np.float32)[None], device=device)
    cap = torch.tensor([float(instance.vehicle_capacities[0])], device=device)
    n = feat.shape[1]
    with torch.no_grad():
        emb, enc, (h, c) = model.encode(feat)
        visited = torch.zeros(1, n, dtype=torch.bool, device=device)
        rem = cap.clone()
        last = torch.tensor([depot], device=device)
        seq: List[int] = []
        max_steps = 3 * n + 2
        for _ in range(max_steps):
            if bool(visited[0, 1:].all()):
                break
            last_emb = emb[torch.arange(1, device=device), last]
            dec_in = torch.cat([last_emb, (rem / cap).unsqueeze(1)], dim=1)
            h, c = model.dec_cell(dec_in, (h, c))
            mask = _feasible_mask(visited, demand, rem, last, depot)
            scores = model.attention(enc, h, mask)
            nxt = int(scores.argmax(dim=1).item())
            seq.append(nxt)
            if nxt == depot:
                rem = cap.clone()
            else:
                visited[0, nxt] = True
                rem = rem - demand[0, nxt]
            last = torch.tensor([nxt], device=device)
    # partir la secuencia en rutas por el depósito
    routes: List[List[int]] = []
    cur: List[int] = []
    for node in seq:
        if node == depot:
            if cur:
                routes.append(cur); cur = []
        else:
            cur.append(node)
    if cur:
        routes.append(cur)
    return routes


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@register_solver("nco-sl")
class NCOSupervised(Solver):
    """Pointer Network supervisada (imita a exact-bc) con inferencia rápida + CRN."""

    def __init__(
        self,
        *,
        train_size: int = 20,
        n_train: int = 512,
        epochs: int = 100,
        hidden: int = 128,
        device: str = "cpu",
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        accident_scale: float = 1.0,
        train_seed: int = 99000,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.train_size = train_size
        self.n_train = n_train
        self.epochs = epochs
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
        return _MODEL_DIR / f"nco_sl_n{self.train_size}_t{self.n_train}_e{self.epochs}_h{self.hidden}.pt"

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        PointerNet = _build_model_cls()
        path = self._model_path()
        if self.cache and path.exists():
            self._model = PointerNet(feat_dim=5, hidden=self.hidden).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            return
        t0 = time.time()
        self._model = train_pointer_net(
            train_size=self.train_size, n_train=self.n_train, epochs=self.epochs,
            hidden=self.hidden, base_seed=self.train_seed, device=self.device,
            verbose=self.verbose,
        )
        self._train_time = time.time() - t0
        if self.cache:
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        self._ensure_model()
        depot = int(instance.metadata.get("depot_index", 0))
        t0 = time.time()
        routes = greedy_routes(self._model, instance, device=self.device, depot=depot)
        infer_time = time.time() - t0  # inferencia rápida (lo que se reporta como runtime)

        R = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=R, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot,
        )
        extras = score.as_extras()
        extras.update({
            "n_routes": len(routes),
            "realizations": R,
            "train_time_s": self._train_time,
            "train_size": self.train_size,
            "n_train": self.n_train,
        })
        return Solution(
            routes=routes,
            total_cost=score.expected_cost,
            runtime=infer_time,
            feasibility=score.feasibility,
            cvr=score.cvr,
            waiting_time=score.waiting_time,
            robustness=score.robustness,
            extras=extras,
        )
