"""Paradigma 5 — EHBG-FACS (la propuesta de la tesis).

Acopla una **Red de Flujo Generativo de Balance Híbrido (HBG)** con un **muestreador de
Colonia de Hormigas (GFACS)** y, opcionalmente, un **módulo epistémico (ENN)**. Implementa
los mecanismos del anteproyecto:

  * **GFlowNet HBG** (Fase 2): el Attention Model (``models.transformer``) parametriza la
    política de avance P_F(s'|s), la de retroceso P_B(s|s') y el flujo de estado F_θ(s),
    además de log Z. Se entrena con un objetivo **híbrido** que combina, de forma
    ponderada por λ_DB ∈ [0.1, 0.9], el **Balance de Trayectoria (TB)** (eq. 1) y el
    **Balance Detallado (DB)** (eq. 2):

        L_TB(τ) = ( log Z + Σ_t log P_F(s_{t+1}|s_t) − Σ_t log P_B(s_t|s_{t+1}) − log R(x) )²
        L_DB(s,s') = ( log F(s) + log P_F(s'|s) − log F(s') − log P_B(s|s') )²
        L_HBG = (1 − λ_DB)·L_TB + λ_DB·L_DB

  * **Recompensa sensible al riesgo** (Fase 1): R(x) ∝ exp(− CVaR_α(c+Q) / T), estimada
    por simulación Monte Carlo CRN (``stochastic.score_routes``). Privilegia rutas robustas
    ante la cola (retrasos log-normales + accidentes de Poisson), no solo de bajo costo
    medio. La energía se normaliza por una referencia (EMA del CVaR del lote) para
    estabilidad numérica; log Z absorbe la constante aditiva.

  * **GFACS** (Fase 3): la matriz heurística a priori η de la GFlowNet siembra un ACO;
    K hormigas construyen rutas muestreando ∝ τ_ACO^α · η^β (con máscara de factibilidad);
    las trayectorias más robustas (menor CVaR) actualizan la feromona. Las soluciones
    exitosas alimentan un **búfer de repetición fuera de política** que retroalimenta el
    entrenamiento de la GFlowNet (TB/DB sobre trayectorias replayed).

  * **ENN (Fase 5, opcional)**: cabeza *epinet* indexada que cuantifica la incertidumbre
    epistémica para guiar la exploración hacia subgrafos poco visitados.

Inferencia: se codifica la instancia, se ejecuta el GFACS guiado por la política
entrenada y se devuelve la mejor solución (menor CVaR). El runner re-puntúa con el
protocolo completo (CRN, R realizaciones), igual que a los demás paradigmas.

Honestidad (limitaciones declaradas): es una implementación **propia** fiel en mecanismo
al anteproyecto y a las líneas de HBG/GFACS/ENN; no es el código oficial de esos papers.
P_B se modela como distribución aprendida sobre el último nodo añadido (DAG de
construcción secuencial), una elección tratable y estándar. La normalización de energía
de la recompensa es una decisión de implementación documentada.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from vrp_bench.core import Instance, Solution, Solver
from .. import stochastic, data as svrp_data
from ..parallel import effective_jobs, process_map
from ..models import transformer as T


# --------------------------------------------------------------------------- #
# Recompensa sensible al riesgo
# --------------------------------------------------------------------------- #


def risk_logreward(cvar: np.ndarray, scale: float, temperature: float) -> np.ndarray:
    """log R(x) = −(CVaR_α / scale) / T. ``scale`` normaliza la energía (estabilidad);
    log Z absorbe el offset, así que la propiedad P_T(x) ∝ R(x) se preserva en escala."""
    return -(np.asarray(cvar, dtype=np.float64) / max(scale, 1e-6)) / temperature


# --------------------------------------------------------------------------- #
# Construcción GFlowNet (on-policy muestreada o forzada para replay off-policy)
# --------------------------------------------------------------------------- #


def construct(model, feat, demand, cap, *, forced=None, sample=True, epi_z=None,
              max_steps=None):
    """Construye trayectorias del MDP del CVRP bajo la GFlowNet.

    Si ``forced`` (LongTensor (B, L), -1 = relleno) se da, sigue esa secuencia (replay
    off-policy); si no, muestrea de P_F (on-policy). Todas las filas corren ``T`` pasos
    fijos (relleno con depósito, transición no-op válida), de modo que los flujos por
    paso se apilan limpios en (T, B).

    Devuelve un dict con tensores apilados para TB/DB y las secuencias resultantes.
    """
    import torch
    device = feat.device
    B, n, _ = feat.shape
    depot = 0
    T_steps = max_steps if max_steps is not None else (3 * n + 2 if forced is None else forced.shape[1])
    emb, graph = model.encode(feat)
    ar = torch.arange(B, device=device)

    visited = torch.zeros(B, n, dtype=torch.bool, device=device)
    rem = cap.clone()
    last = torch.full((B,), depot, dtype=torch.long, device=device)
    logpf_steps, logpb_steps, logF_steps = [], [], []
    seqs = [[] for _ in range(B)]

    for t in range(T_steps):
        last_emb = emb[ar, last]
        mask = T.feasible_mask(visited, demand, rem, last, depot)
        if forced is not None:
            ft = forced[:, t].clamp(min=0)
            mask[ar, ft] = True          # garantiza que la acción forzada sea elegible
        logits, g = model.step_logits(emb, graph, last_emb, rem / cap, mask, epi_z=epi_z)
        logsm = torch.log_softmax(logits, dim=1)

        if forced is not None:
            nxt = forced[:, t].clamp(min=0)
            active = forced[:, t] >= 0
        else:
            probs = torch.softmax(logits, dim=1)
            nxt = torch.multinomial(probs, 1).squeeze(1) if sample else logits.argmax(1)
            active = torch.ones(B, dtype=torch.bool, device=device)

        step_logpf = torch.where(active, logsm[ar, nxt], torch.zeros(B, device=device))
        logF_s = model.flow(g) if model.use_flow else torch.zeros(B, device=device)

        is_depot = nxt == depot
        cust = active & (~is_depot)
        if model.use_backward:
            present = visited.clone(); present[:, depot] = True
            present[ar[cust], nxt[cust]] = True
            blsm = torch.log_softmax(model.backward_logits(emb, present), dim=1)
            step_logpb = torch.where(cust, blsm[ar, nxt], torch.zeros(B, device=device))
        else:
            step_logpb = torch.zeros(B, device=device)

        rem = torch.where(is_depot & active, cap, rem - torch.where(active, demand[ar, nxt], torch.zeros(B, device=device)))
        visited[ar[cust], nxt[cust]] = True
        last = torch.where(active, nxt, last)
        for b in range(B):
            if bool(active[b]) and not bool(is_depot[b]):
                seqs[b].append(int(nxt[b]))
            elif bool(active[b]) and bool(is_depot[b]):
                seqs[b].append(depot)

        logpf_steps.append(step_logpf)
        logpb_steps.append(step_logpb)
        logF_steps.append(logF_s)
        if forced is None and bool(visited[:, 1:].all()):
            # añade unos pasos de cierre al depósito ya cubiertos por relleno
            pass

    return {
        "logpf_steps": torch.stack(logpf_steps, 0),   # (T, B)
        "logpb_steps": torch.stack(logpb_steps, 0),
        "logF_steps": torch.stack(logF_steps, 0),
        "seqs": seqs,
        "B": B,
    }


def hbg_loss(out, logR, log_Z, lam: float, use_flow: bool):
    """Pérdida híbrida HBG = (1−λ)·TB + λ·DB. ``logR`` (B,) tensor."""
    import torch
    logpf = out["logpf_steps"]   # (T,B)
    logpb = out["logpb_steps"]
    logF = out["logF_steps"]

    logpf_sum = logpf.sum(0)     # (B,)
    logpb_sum = logpb.sum(0)
    tb = (log_Z + logpf_sum - logpb_sum - logR) ** 2
    L_tb = tb.mean()

    if use_flow:
        nextF = torch.cat([logF[1:], logR.unsqueeze(0)], dim=0)       # (T,B); último = logR
        db = (logF + logpf - nextF - logpb) ** 2
        L_db = db.mean()
    else:
        L_db = torch.zeros((), device=logpf.device)

    return (1.0 - lam) * L_tb + lam * L_db, float(L_tb.detach()), float(L_db.detach())


# --------------------------------------------------------------------------- #
# GFACS: muestreador de Colonia de Hormigas guiado por la heurística neuronal η
# --------------------------------------------------------------------------- #


def _ant_construct(pher, eta, demand, cap, customers, rng, alpha_aco, beta_aco,
                   epi=None, kappa_epi=0.0):
    """Una hormiga construye una solución (lista de acciones: clientes y retornos al
    depósito), muestreando el siguiente nodo ∝ τ_ACO^α · η^β · (1 + κ·u) con máscara
    de factibilidad. Sin ``epi`` es exactamente la regla del Ant System (Dorigo).

    ``epi`` (n,n) ∈ [0,1] es la **incertidumbre epistémica por arista** (dispersión
    de η entre índices del epinet) y ``kappa_epi`` su peso. El factor (1 + κ·u)
    sesga el muestreo hacia aristas poco exploradas: es el mecanismo que enuncia H3.
    Con κ=0 el término desaparece y se recupera EHBG-FACS base — lo que hace que la
    ablación con/sin ENN sea exacta y no una comparación entre modelos distintos."""
    n = pher.shape[0]
    depot = 0
    visited = np.zeros(n, dtype=bool)
    rem = cap
    cur = depot
    seq = []
    n_cust = len(customers)
    guard = 0
    while visited[1:].sum() < n_cust and guard < 4 * n + 4:
        guard += 1
        fits = (~visited) & (demand <= rem + 1e-6)
        fits[depot] = False
        cand = np.where(fits)[0]
        if cand.size == 0:                       # ninguna factible -> regresar al depósito
            if cur != depot:
                seq.append(depot); cur = depot; rem = cap
            continue
        w = (pher[cur, cand] ** alpha_aco) * (eta[cur, cand] ** beta_aco)
        if epi is not None and kappa_epi:
            w = w * (1.0 + kappa_epi * epi[cur, cand])
        s = w.sum()
        p = w / s if s > 0 else np.ones_like(w) / w.size
        j = int(rng.choice(cand, p=p))
        seq.append(j); visited[j] = True; rem -= demand[j]; cur = j
    return seq


# Contexto heredado por los workers fork del enjambre paralelo (solo numpy; los
# escenarios ξ pre-muestreados pueden pesar GB a n grande y se comparten por
# copy-on-write en vez de serializarse por tarea).
_ANT_CTX: dict = {}


def _ant_task(args):
    it, k, pher = args
    ctx = _ANT_CTX
    rng = np.random.default_rng(np.random.SeedSequence([ctx["seed"], it, k]))
    seq = _ant_construct(pher, ctx["eta"], ctx["demand"], ctx["cap"], ctx["customers"],
                         rng, ctx["alpha_aco"], ctx["beta_aco"],
                         epi=ctx.get("epi"), kappa_epi=ctx.get("kappa_epi", 0.0))
    routes = T.split_routes(seq, 0)
    sc = stochastic.score_routes(
        ctx["instance"], routes, num_realizations=ctx["R"], seed=ctx["inst_seed"],
        alpha=ctx["alpha_cvar"], late_penalty=ctx["late_penalty"],
        accident_scale=ctx["accident_scale"], scenarios=ctx["scens"],
        r_offset=ctx["r_offset"])
    return seq, routes, sc


def aco_search(eta, instance, *, n_ants=12, n_iters=8, alpha_aco=1.0, beta_aco=2.0,
               rho=0.1, Q=1.0, elite=3, aco_realizations=30, seed=0, late_penalty=1.0,
               accident_scale=1.0, alpha_cvar=0.95, n_jobs=1, trace=False,
               r_offset=0, epi=None, kappa_epi=0.0):
    """ACO guiado por η con feromona; puntúa cada hormiga con CVaR (CRN, R reducido).
    Devuelve (mejor_rutas, mejor_score, pool) donde ``pool`` = trayectorias elite para el
    búfer de replay.

    ``r_offset`` — **anti-fuga de escenarios**. La búsqueda puntúa sus candidatas sobre
    ξ con r ∈ [r_offset, r_offset+aco_realizations). Con el valor del protocolo
    (``proto.search_offset`` = R_eval) esos escenarios son **disjuntos** de los de
    evaluación, de modo que elegir el argmin CVaR aquí no es seleccionar sobre la
    muestra con la que después se mide el método.

    Eficiencia (idéntica en resultados): los escenarios ξ se **pre-muestrean una vez**
    (``presample_scenarios``) y se reutilizan para todas las hormigas (antes cada
    puntuación re-muestreaba el mismo ruido CRN). Con ``n_jobs>1`` las hormigas de cada
    iteración se construyen/puntúan en procesos fork paralelos ("enjambre estocástico
    paralelo" del anteproyecto). Reproducibilidad: cada hormiga usa un RNG propio
    derivado de ``SeedSequence([seed, iteración, hormiga])``, de modo que el resultado
    es **independiente de n_jobs y del orden de ejecución**.
    """
    n = eta.shape[0]
    demand = np.asarray(instance.demands, dtype=np.float64)
    cap = float(np.asarray(instance.vehicle_capacities, dtype=np.float64).ravel()[0])
    customers = [i for i in range(n) if i != 0]
    inst_seed = int(instance.metadata.get("seed", seed))

    scens = stochastic.presample_scenarios(n, inst_seed, aco_realizations,
                                           accident_scale=accident_scale,
                                           r_offset=r_offset)
    _ANT_CTX.clear()
    _ANT_CTX.update(dict(
        eta=eta, demand=demand, cap=cap, customers=customers, seed=int(seed),
        instance=instance, R=aco_realizations, inst_seed=inst_seed,
        alpha_cvar=alpha_cvar, late_penalty=late_penalty,
        accident_scale=accident_scale, alpha_aco=alpha_aco, beta_aco=beta_aco,
        scens=scens, r_offset=int(r_offset), epi=epi,
        kappa_epi=float(kappa_epi)))

    jobs = effective_jobs(n_jobs)
    use_pool = (jobs > 1 and n_ants > 1 and not mp.current_process().daemon
                and "fork" in mp.get_all_start_methods())
    ant_pool = (mp.get_context("fork").Pool(processes=min(jobs, n_ants))
                if use_pool else None)

    pher = np.ones((n, n), dtype=np.float64)
    best_routes, best_score, best_cost = None, None, np.inf
    pool = []
    hist = {"best_cvar": [], "iter_best_cvar": [], "iter_mean_cvar": [],
            "unique_ratio": [], "epi_mean": (float(np.mean(epi)) if epi is not None
                                             else float("nan"))}
    try:
        for it in range(n_iters):
            tasks = [(it, k, pher) for k in range(n_ants)]
            ants_raw = ant_pool.map(_ant_task, tasks) if ant_pool else \
                [_ant_task(t) for t in tasks]
            ants = []
            for seq, routes, sc in ants_raw:
                cost = sc.cvar
                ants.append((seq, routes, cost, sc))
                if cost < best_cost:
                    best_cost, best_routes, best_score = cost, routes, sc
            if trace:
                costs_it = [a[2] for a in ants]
                hist["iter_best_cvar"].append(float(min(costs_it)))
                hist["iter_mean_cvar"].append(float(np.mean(costs_it)))
                hist["best_cvar"].append(float(best_cost))
                # diversidad muestral: proporción de soluciones DISTINTAS entre
                # las hormigas de la iteración (diagnóstico anti-colapso de modo)
                hist["unique_ratio"].append(
                    len({tuple(a[0]) for a in ants}) / max(1, len(ants)))
            pher *= (1.0 - rho)                          # evaporación
            for seq, routes, cost, sc in sorted(ants, key=lambda a: a[2])[:elite]:
                dep = Q / (cost + 1e-9)
                prev = 0
                for node in seq:
                    pher[prev, node] += dep; pher[node, prev] += dep
                    prev = node
                pool.append((seq, cost))
    finally:
        if ant_pool is not None:
            ant_pool.close(); ant_pool.join()
        _ANT_CTX.clear()
    if trace:
        return best_routes, best_score, pool, hist
    return best_routes, best_score, pool


# --------------------------------------------------------------------------- #
# Entrenamiento EHBG-FACS
# --------------------------------------------------------------------------- #


def _eta_numpy(model, feat, device):
    """Matriz heurística η (n,n) numpy para una sola instancia (batch 1)."""
    import torch
    with torch.no_grad():
        emb, _ = model.encode(feat)
        eta = model.heuristic_matrix(emb)[0].detach().cpu().numpy()
    return eta


def _eta_numpy_batch(model, feat, device):
    """Matrices heurísticas η (B,n,n) numpy del lote en UNA pasada GPU (antes se
    codificaba instancia por instancia)."""
    import torch
    with torch.no_grad():
        emb, _ = model.encode(feat)
        return model.heuristic_matrix(emb).detach().cpu().numpy()


def _cvar_task(args):
    """Puntuación CRN de una construcción on-policy (worker fork, solo numpy)."""
    inst, routes, R, alpha_cvar, late_penalty, accident_scale = args
    sc = stochastic.score_routes(
        inst, routes, num_realizations=R, seed=int(inst.metadata.get("seed", 0)),
        alpha=alpha_cvar, late_penalty=late_penalty, accident_scale=accident_scale)
    return sc.cvar


def _refine_task(args):
    """Refinamiento GFACS de una instancia del lote (worker fork; ACO secuencial
    dentro del worker — el paralelismo va por instancia)."""
    (eta, inst, aco_realizations, late_penalty, accident_scale, alpha_cvar,
     r_offset) = args
    _, _, pool = aco_search(
        eta, inst, aco_realizations=aco_realizations,
        seed=int(inst.metadata.get("seed", 0)), late_penalty=late_penalty,
        accident_scale=accident_scale, alpha_cvar=alpha_cvar,
        n_ants=8, n_iters=4, n_jobs=1, r_offset=r_offset)
    return sorted(pool, key=lambda a: a[1])[:2]


def train_ehbg_facs(train_sizes=(10, 20), n_train=64, epochs=40, embed_dim=128, n_heads=8,
                    n_layers=3, lr=1e-4, lam_db=0.5, temperature=2.0, batch=16,
                    aco_realizations=30, train_realizations=30, refine_every=5,
                    replay_ratio=0.5, buffer_size=8, epinet=False, base_seed=77000,
                    device="cpu", late_penalty=1.0, accident_scale=1.0, alpha_cvar=0.95,
                    n_jobs=1, verbose=True, epi_prior_scale=1.0, search_r_offset=0):
    """Entrena la GFlowNet HBG con recompensa CVaR + refinamiento GFACS + replay off-policy.

    Banco de entrenamiento **fijo** (semillas disjuntas del banco de evaluación) para que
    el búfer de replay acumule trayectorias elite por instancia a lo largo de las épocas.

    ``n_jobs``: procesos fork para las fases CPU entre pasos de GPU — la puntuación
    CRN del lote on-policy y el refinamiento GFACS por instancia (los workers solo
    usan numpy; la GPU queda para encode/backward).
    """
    import torch
    torch.manual_seed(base_seed)
    model = T.AttentionModel(embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
                             use_flow=True, use_backward=True, use_heuristic=True,
                             epinet=epinet, epi_prior_scale=epi_prior_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Banco de entrenamiento fijo, agrupado por tamaño.
    train_bank = {}
    k = 0
    for s in train_sizes:
        insts = [svrp_data.generate_instance(s, seed=base_seed + 500000 + k + j,
                                             capacity_mode="binding") for j in range(n_train)]
        k += n_train
        train_bank[s] = insts
    buffer: Dict[int, deque] = {}     # id(instancia) -> deque[(forced_seq, cvar)]
    energy_scale = {s: 1.0 for s in train_sizes}      # EMA del CVaR por tamaño (energía)

    history = {"loss": [], "tb": [], "db": [], "cvar": []}
    rng = np.random.default_rng(base_seed)

    for ep in range(epochs):
        for s in train_sizes:
            insts = train_bank[s]
            order = rng.permutation(len(insts))
            for i0 in range(0, len(insts), batch):
                idx = order[i0:i0 + batch]
                binsts = [insts[i] for i in idx]
                feat, demand, cap, _ = T.instance_tensors(binsts, device)
                B = len(binsts)
                epi_z = (torch.randn(B, model.epi_index_dim, device=device) if epinet else None)

                # --- on-policy (puntuación CRN del lote en paralelo, CPU) ---
                out = construct(model, feat, demand, cap, sample=True, epi_z=epi_z)
                cvars = np.asarray(process_map(
                    _cvar_task,
                    [(binsts[b], T.split_routes(out["seqs"][b], 0), train_realizations,
                      alpha_cvar, late_penalty, accident_scale) for b in range(B)],
                    n_jobs=n_jobs))
                energy_scale[s] = 0.9 * energy_scale[s] + 0.1 * float(np.mean(cvars))
                logR = torch.as_tensor(risk_logreward(cvars, energy_scale[s], temperature),
                                       dtype=torch.float32, device=device)
                loss, l_tb, l_db = hbg_loss(out, logR, model.log_Z, lam_db, model.use_flow)

                # --- GFACS refinement -> replay buffer (cada refine_every pasos) ---
                # η del lote en UNA pasada GPU; refinamiento por instancia en paralelo.
                if ep % refine_every == 0:
                    etas = _eta_numpy_batch(model, feat, device)
                    tops = process_map(
                        _refine_task,
                        [(etas[b], binsts[b], aco_realizations, late_penalty,
                          accident_scale, alpha_cvar, search_r_offset)
                         for b in range(B)],
                        n_jobs=n_jobs)
                    for b in range(B):
                        buf = buffer.setdefault(id(binsts[b]), deque(maxlen=buffer_size))
                        for seq, cost in tops[b]:
                            buf.append((seq, cost))

                # --- off-policy replay (TB/DB sobre trayectorias del búfer) ---
                replays = [(b, *buffer[id(binsts[b])][rng.integers(len(buffer[id(binsts[b])]))])
                           for b in range(B) if id(binsts[b]) in buffer and len(buffer[id(binsts[b])]) > 0]
                if replays and replay_ratio > 0:
                    rb = [r[0] for r in replays]
                    seqs = [r[1] for r in replays]
                    rcv = np.array([r[2] for r in replays])
                    Lmax = max(len(sq) for sq in seqs) + 1
                    forced = torch.full((len(rb), Lmax), -1, dtype=torch.long, device=device)
                    for j, sq in enumerate(seqs):
                        forced[j, :len(sq)] = torch.as_tensor(sq, device=device)
                        forced[j, len(sq)] = 0       # cierre al depósito
                    fr = feat[rb]; dr = demand[rb]; cr = cap[rb]
                    epi_zr = (torch.randn(len(rb), model.epi_index_dim, device=device) if epinet else None)
                    out_r = construct(model, fr, dr, cr, forced=forced, epi_z=epi_zr)
                    logR_r = torch.as_tensor(risk_logreward(rcv, energy_scale[s], temperature),
                                             dtype=torch.float32, device=device)
                    loss_r, _, _ = hbg_loss(out_r, logR_r, model.log_Z, lam_db, model.use_flow)
                    loss = loss + replay_ratio * loss_r

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                history["loss"].append(float(loss.detach())); history["tb"].append(l_tb)
                history["db"].append(l_db); history["cvar"].append(float(np.mean(cvars)))
        if verbose and (ep + 1) % 5 == 0:
            print(f"[ehbg-facs] época {ep+1}/{epochs}  L={history['loss'][-1]:.4f} "
                  f"(TB={history['tb'][-1]:.3f} DB={history['db'][-1]:.3f})  "
                  f"CVaR_medio≈{np.mean(history['cvar'][-10:]):.1f}")
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class EHBGFACS(Solver):
    """Muestreo híbrido HBG-GFlowNet + refinamiento poblacional GFACS (propuesta)."""
    name = "ehbg-facs"
    epinet = False

    def __init__(self, *, train_sizes=(10, 20), n_train=64, epochs=40, embed_dim=128,
                 n_heads=8, n_layers=3, lam_db=0.5, temperature=2.0, lr=1e-4,
                 batch=16, refine_every=5, aco_train_realizations=30, train_realizations=30,
                 infer_ants=16, infer_iters=12, infer_realizations=40, alpha_aco=1.0,
                 beta_aco=2.0, rho=0.1, device="cpu", default_realizations=200, alpha=0.95,
                 late_penalty=1.0, accident_scale=1.0, train_seed=77000, models_dir=None,
                 cache=True, n_jobs=1, verbose=True, search_r_offset=None,
                 kappa_epi=0.0, epi_index_samples=8, epi_prior_scale=1.0):
        self.train_sizes = tuple(train_sizes)
        self.n_train = n_train
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lam_db = lam_db
        self.temperature = temperature
        self.lr = lr
        self.batch = batch
        self.refine_every = refine_every
        self.aco_train_realizations = aco_train_realizations
        self.train_realizations = train_realizations
        self.infer_ants = infer_ants
        self.infer_iters = infer_iters
        self.infer_realizations = infer_realizations
        self.alpha_aco = alpha_aco
        self.beta_aco = beta_aco
        self.rho = rho
        self.device = device
        self.default_realizations = default_realizations
        self.alpha = alpha
        self.late_penalty = late_penalty
        self.accident_scale = accident_scale
        self.train_seed = train_seed
        self.models_dir = Path(models_dir) if models_dir else Path.cwd()
        self.cache = cache
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Anti-fuga de escenarios: la búsqueda GFACS puntúa sus candidatas sobre
        # ξ con r >= search_r_offset. Por defecto = default_realizations (= R de
        # evaluación), de modo que búsqueda y evaluación son DISJUNTAS. Pasar 0
        # reproduce el comportamiento sesgado, sólo para cuantificar el sesgo.
        self.search_r_offset = (int(default_realizations) if search_r_offset is None
                                else int(search_r_offset))
        # ENN (H3): peso del bono epistémico en la regla de transición del ACO y
        # nº de índices z con que se estima la dispersión. κ=0 desactiva el bono.
        self.kappa_epi = float(kappa_epi)
        self.epi_index_samples = int(epi_index_samples)
        self.epi_prior_scale = float(epi_prior_scale)
        self._model = None
        self._train_time = 0.0
        self.history = {}

    def _eta_for(self, instance):
        """Devuelve ``(η, u)``: matriz heurística a priori que siembra el ACO y la
        incertidumbre epistémica por arista (``None`` si no hay epinet).

        La clase base la toma de la GFlowNet entrenada; ``FACSDistancePrior`` la
        sustituye por 1/d; ``EHBGFACSEpistemic`` promedia sobre índices del epinet y
        devuelve además su dispersión."""
        feat, _, _, _ = T.instance_tensors([instance], self.device)
        return _eta_numpy(self._model, feat, self.device), None

    def _model_path(self) -> Path:
        sz = "-".join(str(s) for s in self.train_sizes)
        tag = "enn" if self.epinet else "base"
        return self.models_dir / (f"ehbg_facs_{tag}_n{sz}_e{self.epochs}_h{self.embed_dim}"
                                  f"_lam{self.lam_db}_T{self.temperature}.pt")

    def ensure_model(self):
        if self._model is not None:
            return
        import torch
        path = self._model_path()
        if self.cache and path.exists():
            self._model = T.AttentionModel(embed_dim=self.embed_dim, n_heads=self.n_heads,
                                           n_layers=self.n_layers, use_flow=True,
                                           use_backward=True, use_heuristic=True,
                                           epinet=self.epinet,
                                           epi_prior_scale=self.epi_prior_scale).to(self.device)
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            if self.verbose:
                print(f"[ehbg-facs] modelo cargado de cache: {path.name}")
            return
        t0 = time.time()
        self._model, self.history = train_ehbg_facs(
            train_sizes=self.train_sizes, n_train=self.n_train, epochs=self.epochs,
            embed_dim=self.embed_dim, n_heads=self.n_heads, n_layers=self.n_layers,
            lr=self.lr, lam_db=self.lam_db, temperature=self.temperature, batch=self.batch,
            aco_realizations=self.aco_train_realizations,
            train_realizations=self.train_realizations, refine_every=self.refine_every,
            epinet=self.epinet, base_seed=self.train_seed, device=self.device,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale,
            alpha_cvar=self.alpha, n_jobs=self.n_jobs, verbose=self.verbose,
            epi_prior_scale=self.epi_prior_scale,
            search_r_offset=self.search_r_offset)
        self._train_time = time.time() - t0
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        self.ensure_model()
        depot = int(instance.metadata.get("depot_index", 0))
        t0 = time.time()
        eta, epi = self._eta_for(instance)
        routes, _, _, search_hist = aco_search(
            eta, instance, n_ants=self.infer_ants, n_iters=self.infer_iters,
            alpha_aco=self.alpha_aco, beta_aco=self.beta_aco, rho=self.rho,
            aco_realizations=self.infer_realizations,
            seed=int(instance.metadata.get("seed", 0)), late_penalty=self.late_penalty,
            accident_scale=self.accident_scale, alpha_cvar=self.alpha,
            n_jobs=self.n_jobs, trace=True, r_offset=self.search_r_offset,
            epi=epi, kappa_epi=(self.kappa_epi if epi is not None else 0.0))
        infer_time = time.time() - t0
        routes = routes or []

        Rz = num_realizations if num_realizations and num_realizations > 1 else self.default_realizations
        seed = int(instance.metadata.get("seed", 0))
        score = stochastic.score_routes(
            instance, routes, num_realizations=Rz, seed=seed, alpha=self.alpha,
            late_penalty=self.late_penalty, accident_scale=self.accident_scale, depot=depot)
        extras = score.as_extras()
        extras.update({"n_routes": len(routes), "realizations": Rz,
                       "train_time_s": self._train_time, "method": self.name,
                       "lambda_db": self.lam_db, "temperature": self.temperature,
                       "epistemic": self.epinet, "train_sizes": list(self.train_sizes),
                       # Auditoría anti-fuga: rangos de ξ de búsqueda vs evaluación.
                       "search_r_offset": self.search_r_offset,
                       "search_realizations": self.infer_realizations,
                       "search_disjoint": bool(self.search_r_offset >= Rz),
                       "search_budget": int(self.infer_ants * self.infer_iters),
                       "kappa_epi": (self.kappa_epi if epi is not None else 0.0),
                       "epi_mean": search_hist.get("epi_mean", float("nan")),
                       "search_trace": search_hist,
                       "diversity": (float(np.mean(search_hist["unique_ratio"]))
                                     if search_hist["unique_ratio"] else float("nan"))})
        return Solution(routes=routes, total_cost=score.expected_cost, runtime=infer_time,
                        feasibility=score.feasibility, cvr=score.cvr,
                        waiting_time=score.waiting_time, robustness=score.robustness, extras=extras)


class FACSDistancePrior(EHBGFACS):
    """Ablación de atribución (d): **mismo presupuesto, sin red neuronal**.

    Reemplaza la matriz heurística aprendida η por el prior clásico del Ant System
    η_ij = 1/d_ij, y **conserva todo lo demás idéntico** a ``EHBGFACS``: mismo número
    de hormigas e iteraciones, misma regla de transición τ^α·η^β, misma actualización
    de feromona por CVaR, misma puntuación de candidatas sobre los MISMOS escenarios
    de búsqueda disjuntos, misma selección del argmin.

    Para qué sirve. Si ``ehbg-facs`` no supera a ``facs-dist``, la ventaja frente a
    los baselines no proviene de la GFlowNet sino del presupuesto de búsqueda y de la
    selección sensible al riesgo. Es el control que un sinodal pide primero, y el que
    convierte cualquier resultado positivo en atribuible. No entrena ni carga modelo,
    así que corre en CPU y en segundos.
    """
    name = "facs-dist"
    epinet = False

    def ensure_model(self):      # no hay modelo que entrenar ni cargar
        self._model = None
        self._train_time = 0.0

    def _eta_for(self, instance):
        """η = 1/d (prior de distancia del Ant System), normalizada a (0, 1]."""
        locs = np.asarray(instance.locations, dtype=np.float64)
        d = stochastic.euclidean_int_matrix(locs)
        with np.errstate(divide="ignore"):
            eta = 1.0 / np.maximum(d, 1e-9)
        np.fill_diagonal(eta, 0.0)
        mx = float(eta.max()) or 1.0
        return np.clip(eta / mx, 1e-6, 1.0), None


class EHBGFACSEpistemic(EHBGFACS):
    """Extensión epistémica (Fase 5) — H3, implementada de extremo a extremo.

    Tres piezas, las tres necesarias para que H3 sea contrastable:

    1. **Epinet con red a priori congelada** (``models.transformer``), que es la que
       genera incertidumbre antes de ver datos (Osband et al.). Sin ella la
       "incertidumbre" es ruido entrenable que el ajuste puede anular.
    2. **η indexada por z**: la incertidumbre llega a la matriz que siembra el ACO,
       no sólo a los logits de entrenamiento. Se muestrean ``epi_index_samples``
       índices y se usa la **media** como η y la **desviación estándar** como señal
       epistémica por arista.
    3. **Bono explícito de exploración**: el ACO muestrea ∝ τ^α·η^β·(1+κ·u), de modo
       que la incertidumbre *guía* el muestreo. Con ``kappa_epi=0`` el bono se apaga
       y queda exactamente EHBG-FACS base: la ablación con/sin ENN es limpia.
    """
    name = "ehbg-facs-enn"
    epinet = True

    def __init__(self, *args, kappa_epi=0.5, **kw):
        super().__init__(*args, kappa_epi=kappa_epi, **kw)

    def _eta_for(self, instance):
        import torch
        feat, _, _, _ = T.instance_tensors([instance], self.device)
        with torch.no_grad():
            emb, _ = self._model.encode(feat)
            mean, std = self._model.heuristic_ensemble(emb, n_index=self.epi_index_samples)
        eta = mean[0].detach().cpu().numpy()
        u = std[0].detach().cpu().numpy()
        mx = float(u.max())
        u = u / mx if mx > 1e-12 else np.zeros_like(u)   # dispersión normalizada a [0,1]
        return eta, u
