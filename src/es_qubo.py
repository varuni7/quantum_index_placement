#!/usr/bin/env python3
"""QUBO for Elasticsearch's objective Phi with its shipped balance factors (factored form, paper Eq. 23)."""
import os
import sys
import numpy as np
np.infty = getattr(np, "infty", np.inf)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qubo_move_variables import on_node_expr, yv
from qubo_index_placement import QUBO
from paper_qubo_experiment import _add_squared

F_SHARD, F_INDEX, F_WRITE, F_DISK = 0.45, 0.55, 10.0, 2e-11
_S = F_SHARD + F_INDEX + F_WRITE + F_DISK
TH0, TH1, TH2, TH3 = F_SHARD/_S, F_INDEX/_S, F_WRITE/_S, F_DISK/_S


def _merge(*parts):
    acc, c = {}, 0.0
    for terms, const, scale in parts:
        for v, a in terms:
            acc[v] = acc.get(v, 0.0) + scale * a
        c += scale * const
    return [(v, a) for v, a in acc.items() if a], c


def data_expr(inst, P0, data_by, node_name):
    nodes = [n.name for n in inst.usable_nodes()]
    terms, used_cur = [], 0.0
    for rep in inst.replicas:
        a = float(data_by.get(rep.name, 0.0))
        home = P0.get(rep.name)
        if home == node_name:
            used_cur += a
            if a:
                for B in nodes:
                    if B != home:
                        terms.append((yv(rep.name, B), -a))
        elif a:
            terms.append((yv(rep.name, node_name), +a))
    return terms, used_cur


def _add_product(q, t1, c1, t2, c2, w):
    for v1, a1 in t1:
        for v2, a2 in t2:
            if v1 == v2:
                q.add_linear(v1, w * a1 * a2)
            else:
                q.add_quadratic(v1, v2, w * a1 * a2)
    for v1, a1 in t1:
        q.add_linear(v1, w * a1 * c2)
    for v2, a2 in t2:
        q.add_linear(v2, w * a2 * c1)
    q.add_offset(w * c1 * c2)


def build_es_qubo(inst, P0, data_by, wload_by=None, theta=(TH0, TH1, TH2, TH3), scale_disk=True, lam=None,
                  cap_per_node=None, high_watermark=0.90, move_budget=None, slack_bits=10,
                  onedest_mult=1.0,
                  factored=True):
    th0, th1, th2, th3 = theta
    q = QUBO()
    nodes = [n.name for n in inst.usable_nodes()]
    N = len(nodes)
    logicals = sorted({r.logical for r in inst.replicas})
    K = len(logicals)
    byidx = {L: [r for r in inst.replicas if r.logical == L] for L in logicals}

    unit = (sum(data_by.values()) / max(len(data_by), 1)) if scale_disk else 1.0

    if lam is None:
        wl_max = max(wload_by.values()) if wload_by else 0.0
        sz_max = max(data_by.values()) / (unit if scale_disk else 1.0)
        delta_max = th0 + th1 + th2 * wl_max + th3 * sz_max
        lam = 100.0 * max(1.0, K * delta_max * delta_max, delta_max)

    th3 = th3 * unit

    tot_shards = len(inst.replicas)
    avg_shards = tot_shards / N
    tot_disk = sum(data_by.values()) / unit
    avg_disk = tot_disk / N
    use_wl = wload_by is not None and any(wload_by.values())
    avg_wload = (sum(wload_by.values()) / N) if use_wl else 0.0

    if factored:
        _prod = _add_product
        cA = K * th0 * th0 + 2 * th0 * th1
        cD = K * th3 * th3
        cAD = 2 * th3 * (K * th0 + th1)
        cL  = K * th2 * th2
        cAL = 2 * th2 * (K * th0 + th1)
        cLD = 2 * K * th2 * th3
        for nm in nodes:
            a_t, a_c = _merge(*[(*on_node_expr(inst, P0, r, nm), 1.0) for r in inst.replicas])
            a_c -= avg_shards
            d_raw, d_rc = data_expr(inst, P0, data_by, nm)
            d_t = [(v, x / unit) for v, x in d_raw]
            d_c = d_rc / unit - avg_disk
            _add_squared(q, a_t, a_c, cA)
            _add_squared(q, d_t, d_c, cD)
            _prod(q, a_t, a_c, d_t, d_c, cAD)
            if use_wl:
                l_t, l_c = data_expr(inst, P0, wload_by, nm)
                l_c -= avg_wload
                _add_squared(q, l_t, l_c, cL)
                _prod(q, a_t, a_c, l_t, l_c, cAL)
                _prod(q, l_t, l_c, d_t, d_c, cLD)
            for L in logicals:
                b_t, b_c = _merge(*[(*on_node_expr(inst, P0, r, nm), 1.0) for r in byidx[L]])
                _add_squared(q, b_t, b_c - len(byidx[L]) / N, th1 * th1)
    else:
      for nm in nodes:
          sh_terms, sh_c = _merge(*[(*on_node_expr(inst, P0, r, nm), 1.0) for r in inst.replicas])
          d_t, d_c = data_expr(inst, P0, data_by, nm)
          dk_terms, dk_c = _merge((d_t, d_c, 1.0 / unit))
          wl_terms, wl_c = (data_expr(inst, P0, wload_by, nm) if use_wl else ([], 0.0))
          for L in logicals:
              avg_i = len(byidx[L]) / N
              ix_terms, ix_c = _merge(*[(*on_node_expr(inst, P0, r, nm), 1.0) for r in byidx[L]])
              terms, const = _merge(
                  (sh_terms, sh_c - avg_shards, th0),
                  (ix_terms, ix_c - avg_i,      th1),
                  (wl_terms, wl_c - avg_wload,  th2),
                  (dk_terms, dk_c - avg_disk,   th3),
              )
              _add_squared(q, terms, const, 1.0)

    _l1 = (lam if lam is not None else 1.0) * onedest_mult
    for r in inst.replicas:
        t = [(yv(r.name, B), 1.0) for B in nodes if B != P0.get(r.name)]
        if not t:
            continue
        _add_squared(q, t, 0.0, _l1)
        for v, a in t:
            q.add_linear(v, -_l1 * a)

    for L in logicals:
        for nm in nodes:
            t, c = _merge(*[(*on_node_expr(inst, P0, r, nm), 1.0) for r in byidx[L]])
            _add_squared(q, t, c, lam)
            for v, a in t:
                q.add_linear(v, -lam * a)
            q.add_offset(-lam * c)
    if cap_per_node:
        limit = high_watermark * cap_per_node / unit
        g = max(limit / (2 ** slack_bits - 1), 1e-9)
        for nm in nodes:
            t, c = data_expr(inst, P0, data_by, nm)
            terms = [(v, a / unit) for v, a in t]
            terms += [(f"cs|{nm}|{b}", g * (2 ** b)) for b in range(slack_bits)]
            _add_squared(q, terms, c / unit - limit, lam)

    if move_budget:
        wtot = sum(data_by.values()) / unit
        cap_mv = move_budget * wtot
        g = max(cap_mv / (2 ** slack_bits - 1), 1e-9)
        mterms = []
        for r in inst.replicas:
            for B in nodes:
                if B != P0.get(r.name):
                    mterms.append((yv(r.name, B), data_by[r.name] / unit))
        mterms += [(f"ms|{b}", g * (2 ** b)) for b in range(slack_bits)]
        _add_squared(q, mterms, -cap_mv, lam)

    return q, dict(K=K, N=N, unit=unit, lam=lam)
