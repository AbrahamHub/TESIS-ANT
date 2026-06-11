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
    resto de paradigmas.

Aclaraciones de la revisión:
  * **P1/P2 — maestro configurable.** SVRPBench **no** trae un baseline de NCO
    *supervisado* (sus baselines de aprendizaje son RL/POMO = paradigma 4), así que esta
    es una implementación **propia**, no un envoltorio oficial. El NCO **imita al maestro
    que se le indique**: con `exact-bc` (default, `nco-sl`) hereda su fragilidad ante
    ventanas (feas≈0); con un maestro factible (`aco`, variante `nco-sl-feas`) imita rutas
    factibles. Es decir, la (in)factibilidad la define el maestro, no el paradigma NCO.
  * **P3** — la secuencia objetivo se canonicaliza (orden de rutas y orientación) para
    mitigar el problema de óptimos múltiples del aprendizaje supervisado.
  * **P4** — entrena en una **distribución de tamaños** (default 10 y 20) para mejorar la
    generalización; se reporta una pérdida de **validación** (hold-out, P6).
  * **P6** — limitaciones del preliminar: decodificación voraz (sin *beam*), CPU.

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


def teacher_sequence(routes: List[List[int]], locations: np.ndarray, depot: int = 0) -> List[int]:
    """Aplana rutas (clientes) en la secuencia objetivo del decodificador con un
    **orden canónico** (P3) para dar un objetivo consistente entre instancias y mitigar
    el problema de óptimos múltiples:

      * cada ruta se orienta para empezar por el extremo más cercano al depósito;
      * las rutas se ordenan por el ángulo de su primer cliente respecto al depósito.

    ``[[1,2],[3]] -> [.., depot, .., depot]`` (el decodificador arranca en el depósito)."""
    import math
    locs = np.asarray(locations, dtype=np.float64)
    dep = locs[depot]
    canon = []
    for r in routes:
        r = [int(c) for c in r]
        if len(r) >= 2:
            d0 = np.hypot(*(locs[r[0]] - dep))
            d1 = np.hypot(*(locs[r[-1]] - dep))
            if d1 < d0:
                r = r[::-1]
        canon.append(r)
    canon.sort(key=lambda r: math.atan2(*(locs[r[0]] - dep)[::-1]))
    seq: List[int] = []
    for r in canon:
        seq.extend(r)
        seq.append(depot)
    return seq


def _make_teacher(name: str):
    """Solver maestro que genera las etiquetas (P1, configurable)."""
    if name == "exact-bc":
        from .exact_bc import ExactBranchCut
        return ExactBranchCut(time_limit=20.0, default_realizations=1)
    if name == "exact-bc-tw":
        from .exact_bc_tw import ExactBranchCutTW
        return ExactBranchCutTW(time_limit=20.0, default_realizations=1)
    if name == "aco":
        from .metaheuristic import ACO
        return ACO(default_realizations=1, n_seeds=1)
    raise ValueError(f"maestro desconocido: {name!r}")


# --------------------------------------------------------------------------- #
# Modelo (definido perezosamente para no exigir torch al importar el paquete)
# --------------------------------------------------------------------------- #


def _build_model_cls():
    import torch
    import torch.nn as nn

    class PointerNet(nn.Module):
        def __init__(self, feat_dim: int = 5, hidden: int = 128, dropout: float = 0.1):
            super().__init__()
            self.hidden = hidden
            self.embed = nn.Linear(feat_dim, hidden)
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.dec_cell = nn.LSTMCell(hidden + 1, hidden)
            self.drop = nn.Dropout(dropout)  # N5: regularización (off en eval -> inferencia determinista)
            # Atención aditiva (Bahdanau): score = v^T tanh(W1 enc + W2 query)
            self.W1 = nn.Linear(hidden, hidden, bias=False)
            self.W2 = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)

        def encode(self, feat):  # feat (B, n, F)
            emb = self.drop(self.embed(feat))       # (B, n, H)
            enc, (h, c) = self.encoder(emb)         # enc (B, n, H)
            return emb, self.drop(enc), (h[-1], c[-1])

        def attention(self, enc, query, mask):      # enc (B,n,H), query (B,H), mask (B,n) bool feasible
            w1 = self.W1(enc)                        # (B,n,H)
            w2 = self.W2(self.drop(query)).unsqueeze(1)  # (B,1,H)
            scores = self.v(torch.tanh(w1 + w2)).squeeze(-1)  # (B,n)
            scores = scores.masked_fill(~mask, float("-inf"))
            return scores

    return PointerNet


# --------------------------------------------------------------------------- #
# Entrenamiento supervisado
# --------------------------------------------------------------------------- #


def _gen_labeled(train_sizes, n_per_size: int, base_seed: int, teacher_name: str):
    """Genera instancias de entrenamiento (de varios tamaños, P4) y sus etiquetas del
    maestro ``teacher_name`` (P1). Las semillas son disjuntas de las de test."""
    teacher = _make_teacher(teacher_name)
    data = []
    k = 0
    for size in train_sizes:
        for _ in range(n_per_size):
            inst = svrp_io.generate_instance(size, seed=base_seed + k, capacity_mode="binding")
            k += 1
            sol = teacher.solve(inst, num_realizations=1)
            if not sol.routes:
                continue
            feat = node_features(inst)
            seq = teacher_sequence(sol.routes, inst.locations, depot=0)
            data.append((inst, feat, seq))
    return data


def train_pointer_net(train_sizes=(10, 20), n_per_size: int = 256, epochs: int = 100,
                      hidden: int = 128, dropout: float = 0.1, lr: float = 1e-3, batch_size: int = 32,
                      base_seed: int = 99000, teacher: str = "exact-bc",
                      val_frac: float = 0.1, device: str = "cpu", verbose: bool = True):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(base_seed)
    PointerNet = _build_model_cls()
    model = PointerNet(feat_dim=5, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    data = _gen_labeled(train_sizes, n_per_size, base_seed, teacher)
    if verbose:
        print(f"[nco-sl] etiquetas ({teacher}): {len(data)} instancias, tamaños {list(train_sizes)}")

    # Agrupar por nº de nodos (P4: multi-tamaño); cada grupo se procesa en lotes de
    # tamaño homogéneo. Pequeño hold-out de validación (P6).
    from collections import defaultdict
    by_n = defaultdict(list)
    for d in data:
        by_n[d[0].num_nodes].append(d)

    def make_tensors(items):
        feats = torch.tensor(np.stack([d[1] for d in items]), device=device)
        demands = torch.tensor(np.stack([np.asarray(d[0].demands, np.float32) for d in items]), device=device)
        caps = torch.tensor([float(d[0].vehicle_capacities[0]) for d in items], device=device)
        seqs = [d[2] for d in items]
        max_len = max(len(s) for s in seqs)
        targets = torch.full((len(items), max_len), -100, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            targets[i, :len(s)] = torch.tensor(s, device=device)
        return feats, demands, caps, targets

    train_groups, val_groups = [], []
    for n_nodes, items in by_n.items():
        n_val = max(1, int(val_frac * len(items)))
        val_groups.append(make_tensors(items[:n_val]))
        train_groups.append(make_tensors(items[n_val:]))

    def epoch_loss(groups, train: bool):
        total, count = 0.0, 0
        for feats, demands, caps, targets in groups:
            N = feats.shape[0]
            perm = torch.randperm(N, device=device) if train else torch.arange(N, device=device)
            for b0 in range(0, N, batch_size):
                idx = perm[b0:b0 + batch_size]
                if train:
                    loss = _seq_loss(model, feats[idx], demands[idx], caps[idx], targets[idx], device, F)
                    opt.zero_grad(); loss.backward(); opt.step()
                else:
                    with torch.no_grad():
                        loss = _seq_loss(model, feats[idx], demands[idx], caps[idx], targets[idx], device, F)
                total += float(loss) * len(idx); count += len(idx)
        return total / max(1, count)

    for ep in range(epochs):
        model.train()
        tr = epoch_loss(train_groups, train=True)
        if verbose and (ep + 1) % 10 == 0:
            model.eval()
            print(f"[nco-sl] epoch {ep+1}/{epochs}  train={tr:.4f}  val={epoch_loss(val_groups, False):.4f}")
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
        teacher: str = "exact-bc",
        train_sizes=(10, 20),
        n_per_size: int = 256,
        epochs: int = 100,
        hidden: int = 128,
        dropout: float = 0.1,
        device: str = "cpu",
        default_realizations: int = 200,
        alpha: float = 0.95,
        late_penalty: float = 1.0,
        accident_scale: float = 1.0,
        train_seed: int = 99000,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.teacher = teacher
        self.train_sizes = tuple(train_sizes)
        self.n_per_size = n_per_size
        self.epochs = epochs
        self.hidden = hidden
        self.dropout = dropout
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
        return _MODEL_DIR / (f"nco_sl_{self.teacher}_n{sz}_t{self.n_per_size}"
                             f"_e{self.epochs}_h{self.hidden}_d{self.dropout}.pt")

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        PointerNet = _build_model_cls()
        path = self._model_path()
        if self.cache and path.exists():
            self._model = PointerNet(feat_dim=5, hidden=self.hidden, dropout=self.dropout).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            return
        t0 = time.time()
        self._model = train_pointer_net(
            train_sizes=self.train_sizes, n_per_size=self.n_per_size, epochs=self.epochs,
            hidden=self.hidden, dropout=self.dropout, base_seed=self.train_seed,
            teacher=self.teacher, device=self.device, verbose=self.verbose,
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
            "teacher": self.teacher,
            "train_sizes": list(self.train_sizes),
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


@register_solver("nco-sl-feas")
class NCOSupervisedFeasible(NCOSupervised):
    """Variante de control (P1): misma Pointer Network supervisada pero **imitando un
    maestro factible** (`aco`) en vez de `exact-bc`. Sirve para separar la limitación
    del *paradigma* NCO de la elección de maestro: si el NCO entrenado con etiquetas
    factibles es factible, la infactibilidad de `nco-sl` proviene de su maestro, no del NCO."""

    def __init__(self, **kw):
        kw.setdefault("teacher", "aco")
        super().__init__(**kw)
