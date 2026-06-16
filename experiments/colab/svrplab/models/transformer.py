"""Attention Model (codificador-decodificador tipo Transformer) para ruteo, en
PyTorch y acelerado por GPU. Es el cuerpo compartido por:

  * Paradigma 3 (NCO supervisado): se entrena por imitación (teacher forcing).
  * Paradigma 4 (NCO por RL): se entrena por REINFORCE estilo POMO.
  * Paradigma 5 (EHBG-FACS): parametriza la política de avance P_F, la de retroceso
    P_B, el flujo de estado F_θ(s) y una matriz heurística a priori η para el ACO.

Fidelidad a la literatura:
  * Codificador: ``L`` capas Transformer (atención multi-cabeza + FF) sobre las
    características de los nodos, **sin codificación posicional** (el conjunto de
    clientes es permutación-invariante), como en Kool et al. (AM) y Kwon et al. (POMO).
  * Decodificador autorregresivo: contexto = (embedding del grafo, último nodo,
    capacidad restante) → *glimpse* de atención multi-cabeza enmascarada → logits de
    compatibilidad de una sola cabeza con recorte tanh (C=10) y máscara de
    factibilidad. Es la arquitectura del "Graph Transformer modificado" de la Fase 2.

MDP de construcción del CVRP (compartido):
  estado = (conjunto visitado, nodo actual, capacidad restante); acción = siguiente
  nodo elegible (cliente que cabe, o el depósito que reinicia la capacidad). Estado
  terminal = todos los clientes visitados.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

FEAT_DIM = 6   # [x/1000, y/1000, demanda/cap, tw_open/1440, tw_close/1440, es_depósito]


def node_features(instance, depot: int = 0) -> np.ndarray:
    """Características estáticas por nodo, normalizadas (independientes de la escala)."""
    locs = np.asarray(instance.locations, dtype=np.float32)
    n = locs.shape[0]
    dem = np.asarray(instance.demands, dtype=np.float32)
    cap = float(np.asarray(instance.vehicle_capacities, dtype=np.float32).ravel()[0])
    span = float(max(locs.max(), 1.0))
    feat = np.zeros((n, FEAT_DIM), dtype=np.float32)
    feat[:, 0] = locs[:, 0] / span
    feat[:, 1] = locs[:, 1] / span
    feat[:, 2] = dem / max(cap, 1.0)
    if instance.time_windows is not None:
        tw = np.asarray(instance.time_windows, dtype=np.float32)
        feat[:, 3] = tw[:, 0] / 1440.0
        feat[:, 4] = tw[:, 1] / 1440.0
    feat[depot, 5] = 1.0
    return feat


def feasible_mask(visited, demand, rem, last, depot: int = 0):
    """(B, n) bool: nodos elegibles. Clientes no visitados que caben; el depósito solo
    si NO estamos ya en él. Si ningún cliente cabe, se fuerza el depósito."""
    import torch
    fits = demand <= (rem.unsqueeze(1) + 1e-4)
    mask = (~visited) & fits
    mask[:, depot] = (last != depot)
    none_feasible = mask[:, 1:].any(dim=1) == False  # noqa: E712
    mask[none_feasible, depot] = True
    return mask


def _build():
    import torch
    import torch.nn as nn

    class AttentionModel(nn.Module):
        """Codificador-decodificador con atención. Expone, además de la política de
        avance, las cabezas que necesita la GFlowNet (flujo, retroceso, heurística)."""

        def __init__(self, feat_dim: int = FEAT_DIM, embed_dim: int = 128,
                     n_heads: int = 8, n_layers: int = 3, ff_dim: int = 512,
                     clip: float = 10.0, use_flow: bool = False,
                     use_backward: bool = False, use_heuristic: bool = False,
                     epinet: bool = False, epi_index_dim: int = 8):
            super().__init__()
            self.embed_dim = embed_dim
            self.n_heads = n_heads
            self.clip = clip
            self.use_flow = use_flow
            self.use_backward = use_backward
            self.use_heuristic = use_heuristic
            self.epinet = epinet
            self.epi_index_dim = epi_index_dim

            self.embed = nn.Linear(feat_dim, embed_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)

            # Decodificador: contexto (grafo, último nodo, capacidad) -> query
            self.ctx = nn.Linear(2 * embed_dim + 1, embed_dim)
            self.glimpse = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # claves de compatibilidad

            # Cabeza de flujo F_θ(s) (DB) y partición log Z (TB) — GFlowNet.
            if use_flow:
                self.flow_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
                self.log_Z = nn.Parameter(torch.zeros(1))
            # Política de retroceso P_B(s|s'): puntúa "qué nodo se añadió al final".
            if use_backward:
                self.back_head = nn.Linear(embed_dim, embed_dim, bias=False)
            # Cabeza heurística η (matriz a priori de aristas para el ACO).
            if use_heuristic:
                self.heur_q = nn.Linear(embed_dim, embed_dim, bias=False)
                self.heur_k = nn.Linear(embed_dim, embed_dim, bias=False)
            # Epinet (ENN): perturbación indexada de la capa final para incertidumbre.
            if epinet:
                self.epi = nn.Sequential(
                    nn.Linear(embed_dim + epi_index_dim, embed_dim), nn.GELU(),
                    nn.Linear(embed_dim, embed_dim))

        # ---- codificación ---------------------------------------------------
        def encode(self, feat):
            """feat (B,n,F) -> (emb (B,n,H), graph (B,H))."""
            emb = self.encoder(self.embed(feat))
            return emb, emb.mean(dim=1)

        # ---- un paso del decodificador -------------------------------------
        def step_logits(self, emb, graph, last_emb, rem_cap, mask, *, epi_z=None):
            """Devuelve los logits de la política de avance sobre los nodos (B,n)."""
            import torch
            q = self.ctx(torch.cat([graph, last_emb, rem_cap.unsqueeze(1)], dim=1))  # (B,H)
            attn_mask = ~mask  # True = posición prohibida en MultiheadAttention
            g, _ = self.glimpse(q.unsqueeze(1), emb, emb,
                                key_padding_mask=attn_mask, need_weights=False)
            g = g.squeeze(1)                                   # (B,H)
            if self.epinet and epi_z is not None:
                g = g + self.epi(torch.cat([g.detach(), epi_z], dim=1))
            k = self.k_proj(emb)                               # (B,n,H)
            logits = torch.einsum("bh,bnh->bn", g, k) / (self.embed_dim ** 0.5)
            logits = self.clip * torch.tanh(logits)
            logits = logits.masked_fill(~mask, float("-inf"))
            return logits, g

        def flow(self, g):
            """log F_θ(s) a partir del contexto del decodificador (B,)."""
            return self.flow_head(g).squeeze(-1)

        def backward_logits(self, emb, present_mask):
            """Logits de P_B(s|s') sobre los nodos presentes en s' (B,n)."""
            import torch
            scores = (self.back_head(emb) * emb).sum(-1) / (self.embed_dim ** 0.5)
            return scores.masked_fill(~present_mask, float("-inf"))

        def heuristic_matrix(self, emb):
            """Matriz heurística a priori η (B,n,n) ≥ 0 para el ACO (GFACS)."""
            import torch
            q = self.heur_q(emb); k = self.heur_k(emb)
            logits = torch.einsum("bih,bjh->bij", q, k) / (self.embed_dim ** 0.5)
            return torch.sigmoid(self.clip * torch.tanh(logits)) + 1e-6

    return AttentionModel


_AM_CLS = None


def AttentionModel(*args, **kwargs):
    """Construye una instancia del Attention Model (clase creada perezosamente para no
    exigir torch al importar el paquete)."""
    global _AM_CLS
    if _AM_CLS is None:
        _AM_CLS = _build()
    return _AM_CLS(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Utilidades de tensores de instancia (compartidas por los rollouts)
# --------------------------------------------------------------------------- #


def instance_tensors(instances, device):
    """Empaqueta una lista de instancias (mismo n) en tensores GPU.
    Devuelve feat (B,n,F), demand (B,n), cap (B,), tau (B,n,n) tiempo nominal."""
    import torch
    from ..stochastic import euclidean_int_matrix, nominal_time_matrix, representative_time
    feat = np.stack([node_features(i) for i in instances])
    demand = np.stack([np.asarray(i.demands, np.float32) for i in instances])
    cap = np.array([float(i.vehicle_capacities[0]) for i in instances], np.float32)
    tau = np.stack([nominal_time_matrix(euclidean_int_matrix(i.locations),
                                        representative_time(i)).astype(np.float32)
                    for i in instances])
    return (torch.as_tensor(feat, device=device),
            torch.as_tensor(demand, device=device),
            torch.as_tensor(cap, device=device),
            torch.as_tensor(tau, device=device))


def split_routes(seq, depot: int = 0):
    """Convierte una secuencia plana (con retornos al depósito) en lista de rutas."""
    routes, cur = [], []
    for node in seq:
        if node == depot:
            if cur:
                routes.append(cur); cur = []
        else:
            cur.append(int(node))
    if cur:
        routes.append(cur)
    return routes
