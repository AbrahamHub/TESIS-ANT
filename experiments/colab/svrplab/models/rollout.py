"""Rollouts del MDP de construcción del CVRP sobre el Attention Model.

Comparte dos modos:
  * ``pomo_rollout`` — N trayectorias por instancia (inicio en nodos distintos),
    muestreadas, con log-prob, costo nominal y entropía (paradigma 4, REINFORCE/POMO).
  * ``greedy_decode`` — decodificación voraz multi-start → mejor ruta por instancia
    (inferencia de los paradigmas 3 y 4).

Ambos son vectorizados sobre el lote (B·N) y corren en GPU. La construcción GFlowNet
(con soporte de secuencia forzada para el replay off-policy) vive en
``solvers.ehbg_facs`` porque necesita lógica específica de EHBG-FACS.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .transformer import feasible_mask, split_routes


def _expand(emb, graph, N):
    """Replica embeddings/graph para N inicios por instancia -> (B*N, ...)."""
    import torch
    B = emb.shape[0]
    idx = torch.arange(B, device=emb.device).repeat_interleave(N)
    return emb[idx], graph[idx], idx


def pomo_rollout(model, feat, demand, cap, tau, *, sample=True, n_starts=None):
    """REINFORCE/POMO: devuelve (logp, cost, entropy) en forma (B, N).

    El primer nodo se **fuerza** (un cliente distinto por trayectoria); su log-prob NO
    se acumula (no es decisión de la política). Se acumula la entropía por paso para
    regularizar (mitiga el colapso de modo)."""
    import torch
    device = feat.device
    B, n, _ = feat.shape
    N = (n - 1) if n_starts is None else n_starts
    depot = 0
    emb, graph = model.encode(feat)
    emb_e, graph_e, bidx = _expand(emb, graph, N)
    BN = B * N
    demand_e = demand[bidx]; cap_e = cap[bidx]
    ar = torch.arange(BN, device=device)

    visited = torch.zeros(BN, n, dtype=torch.bool, device=device)
    rem = cap_e.clone()
    last = torch.full((BN,), depot, dtype=torch.long, device=device)
    logp = torch.zeros(BN, device=device)
    entropy = torch.zeros(BN, device=device)
    cost = torch.zeros(BN, device=device)
    start = torch.arange(N, device=device).repeat(B) + 1   # clientes 1..n-1

    for t in range(3 * n + 2):
        if bool(visited[:, 1:].all()):
            break
        last_emb = emb_e[ar, last]
        mask = feasible_mask(visited, demand_e, rem, last, depot)
        logits, _ = model.step_logits(emb_e, graph_e, last_emb, rem / cap_e, mask)
        logsm = torch.log_softmax(logits, dim=1)
        if t == 0:
            nxt = start
        else:
            if sample:
                nxt = torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1)
            else:
                nxt = logits.argmax(1)
            logp = logp + logsm[ar, nxt]
            p = torch.softmax(logits, 1)
            entropy = entropy - (p * torch.nan_to_num(logsm, neginf=0.0)).sum(1)
        cost = cost + tau[bidx, last, nxt]
        is_depot = nxt == depot
        rem = torch.where(is_depot, cap_e, rem - demand_e[ar, nxt])
        cust = ~is_depot
        visited[ar[cust], nxt[cust]] = True
        last = nxt
    cost = cost + tau[bidx, last, depot]
    return logp.view(B, N), cost.view(B, N), entropy.view(B, N)


def greedy_decode(model, feat, demand, cap, tau, depot: int = 0) -> List[List[List[int]]]:
    """Decodificación voraz multi-start (POMO): devuelve, por instancia, las rutas de
    la mejor trayectoria (menor costo nominal). Lista de longitud B."""
    import torch
    device = feat.device
    B, n, _ = feat.shape
    N = n - 1
    emb, graph = model.encode(feat)
    emb_e, graph_e, bidx = _expand(emb, graph, N)
    BN = B * N
    demand_e = demand[bidx]; cap_e = cap[bidx]
    ar = torch.arange(BN, device=device)

    visited = torch.zeros(BN, n, dtype=torch.bool, device=device)
    rem = cap_e.clone()
    last = torch.full((BN,), depot, dtype=torch.long, device=device)
    cost = torch.zeros(BN, device=device)
    start = torch.arange(N, device=device).repeat(B) + 1
    seqs = [[] for _ in range(BN)]
    with torch.no_grad():
        for t in range(3 * n + 2):
            if bool(visited[:, 1:].all()):
                break
            last_emb = emb_e[ar, last]
            mask = feasible_mask(visited, demand_e, rem, last, depot)
            logits, _ = model.step_logits(emb_e, graph_e, last_emb, rem / cap_e, mask)
            nxt = start if t == 0 else logits.argmax(1)
            cost = cost + tau[bidx, last, nxt]
            is_depot = nxt == depot
            rem = torch.where(is_depot, cap_e, rem - demand_e[ar, nxt])
            for bn in range(BN):
                seqs[bn].append(depot if bool(is_depot[bn]) else int(nxt[bn]))
            cust = ~is_depot
            visited[ar[cust], nxt[cust]] = True
            last = nxt
        cost = cost + tau[bidx, last, depot]
    cost = cost.view(B, N)
    best = cost.argmin(1)                       # mejor inicio por instancia
    out = []
    for b in range(B):
        out.append(split_routes(seqs[b * N + int(best[b])], depot))
    return out
