#!/usr/bin/env python3
"""Couchbase rebalancing: the planner (restarted until the budget ends, best kept) and the QUBO."""
import os, sys, math, time, json
import numpy as np
np.infty = getattr(np, "infty", np.inf)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as AT, bench_full as BF, solvers_bench as S
import generators.cbindexplan_io as cx
from qubo_move_variables import build_move_qubo, decode_move

CHECKPOINTS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120, 300]
# Fixed in advance (gamma_hat in the paper), not tuned per instance.
GHATS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def pareto_add(front, mv, U):
    for (m2, u2) in front:
        if m2 <= mv + 1e-12 and u2 <= U + 1e-12:
            return
    front[:] = [(m2, u2) for (m2, u2) in front
                if not (mv <= m2 + 1e-12 and U <= u2 + 1e-12)]
    front.append((mv, U))


_SCORER = None


def set_scorer(fn):
    global _SCORER
    _SCORER = fn


def cb_parts(pl, nodes, data_by, P0, tot):
    if _SCORER is None:
        raise RuntimeError("call set_scorer first")
    return _SCORER(pl)


def snap(tr, t0, prev, budget):
    el = time.perf_counter() - t0
    while tr["next"] < len(CHECKPOINTS) and CHECKPOINTS[tr["next"]] <= el:
        cp = CHECKPOINTS[tr["next"]]
        if cp <= budget:
            tr["pts"].append((cp, prev[0], prev[1], prev[2]))
        tr["next"] += 1


def run_planner(plan, nodes, data_by, P0, tot, budget, tag, eject=None):
    tr = {"pts": [], "next": 0}
    best = [float("inf"), float("nan"), float("nan")]
    prev = list(best)
    t0, k, tfirst = time.perf_counter(), 0, float("nan")
    front = []
    p0p = os.path.join(BF.TMP, f"cbc_{tag}_0.json")
    p1p = os.path.join(BF.TMP, f"cbc_{tag}_1.json")
    cx.write_plan(p0p, plan)
    while time.perf_counter() - t0 < budget:
        if os.path.exists(p1p):
            os.remove(p1p)
        # -ejectNode tells cbindexplan that a node is leaving.
        cx.run(["-command=rebalance", f"-plan={p0p}", f"-output={p1p}",
                "-enableShardAffinity=false"]
               + ([f"-ejectNode={eject}"] if eject else []), timeout=120)
        k += 1
        snap(tr, t0, prev, budget)
        if os.path.exists(p1p):
            try:
                pj = json.load(open(p1p))
                P1 = {str(i["instId"]): nd["nodeId"]
                      for nd in pj["placement"] for i in (nd.get("indexes") or [])}
                if len(P1) == len(P0):
                    c, U, mv = cb_parts(P1, nodes, data_by, P0, tot)
                    pareto_add(front, mv, U)
                    if c < best[0]:
                        best = [c, U, mv]
                        if math.isnan(tfirst):
                            tfirst = time.perf_counter() - t0
            except Exception:
                pass
        prev = list(best)
    snap(tr, t0, prev, budget)
    return tr["pts"], dict(runs=k, t_best=tfirst, front=sorted(front))


def run_qubo(inst, P0, data_by, mem_by, nodes, tot, budget):
    tr = {"pts": [], "next": 0}
    best = [float("inf"), float("nan"), float("nan")]
    prev = list(best)
    t0, nt, nf, tfirst = time.perf_counter(), 0, 0, float("nan")
    nraw = 0
    front = []
    share = budget / len(GHATS)
    B = BF.penalty_weight_bound(inst)
    dmin = max(1.0, min(v for v in data_by.values() if v > 0))
    w = {k: v / dmin for k, v in data_by.items()}
    for gi, gh in enumerate(GHATS):
        deadline = t0 + share * (gi + 1)
        if time.perf_counter() >= deadline:
            continue
        # lambda = B + 1, beta = 5B, kappa = 1, nu = 20, omega = {d: 3, m: 1, r: 1}
        q = build_move_qubo(inst, P0, lam=B + 1.0, gamma=gh * B, w_move=w,
                            capacity_mode="unbalanced", kappa=1.0, beta=AT.BHAT * B,
                            move_quad=True, onedest_mult=20.0,
                            res_weights={"d": 3.0, "m": 1.0, "r": 1.0})[0]
        case = dict(inst=inst, P0=P0, data_by=data_by, mem_by=mem_by, q=q, eject=None,
                    names=nodes)
        vl, M, off = S.to_dense(q)
        b = 0
        while time.perf_counter() < deadline:
            sm, _, _ = S.s_neal(vl, M, off, q, case, reads=200, sweeps=400, seed=b)
            b += 1
            for x in sm:
                nt += 1
                pl, conf = decode_move({v: int(x[i]) for i, v in enumerate(vl)}, inst, P0)
                if conf or not S.evaluate(x, vl, case, M, off)["feasible"]:
                    continue                      # discarded, not repaired
                nraw += 1
                nf += 1
                c, U, mv = cb_parts(pl, nodes, data_by, P0, tot)
                pareto_add(front, mv, U)
                if c < best[0]:
                    best = [c, U, mv]
                    if math.isnan(tfirst):
                        tfirst = time.perf_counter() - t0
            snap(tr, t0, prev, budget)
            prev = list(best)
    snap(tr, t0, prev, budget)
    return tr["pts"], dict(feas=100.0 * nf / max(nt, 1),
                           raw=100.0 * nraw / max(nt, 1), t_best=tfirst,
                           front=sorted(front))
