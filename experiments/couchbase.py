"""Couchbase experiments (Section 7): fresh placement, rebalancing, bounding movement."""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("CBINDEXPLAN_BIN", os.path.join(ROOT, "bin", "cbindexplan"))

import numpy as np
np.infty = getattr(np, "infty", np.inf)

CHECKPOINTS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120, 300]


def _planner_bin():
    b = os.environ["CBINDEXPLAN_BIN"]
    if not os.path.exists(b):
        raise SystemExit(f"cbindexplan not found at {b}. Build it from Couchbase's repository and set "
                         f"CBINDEXPLAN_BIN to its path (see README), or run with --qubo-only.")
    return b


def fresh_cell(k, skew, seed, nodes=6, budget=300.0, qubo_only=False):
    import anytime as AT, bench_full as BF, solvers_bench as S
    import generators.cbindexplan_io as cx
    from exp_cbcost import snap
    from cb_scenario1 import make_s1
    from cb_cost import cb_cost
    from qubo_move_variables import build_move_qubo, decode_move
    from cqm_experiment import place_feasible

    inst, anchor, res, nodes_l, plan = make_s1(k, seed, skew, n_nodes=nodes)
    res["scan"] = {r: 0.0 for r in res["scan"]}
    for nd in plan["placement"]:
        for i in (nd.get("indexes") or []):
            i["actualScanRate"] = 0
        nd["actualScanRate"] = 0
    for r in inst.replicas:
        r.m = 0.0
    score = lambda pl: cb_cost(pl, nodes_l, res, pl, set(), ())[0]
    nres = len(cb_cost(anchor, nodes_l, res, anchor, set(), ())[1]["cvs"])
    out = dict(experiment="couchbase_fresh", k=k, skew=skew, seed=seed, nodes=nodes,
               budget=budget, anchor=score(anchor), n_resources=nres)

    if not qubo_only:
        BIN = _planner_bin()
        byname = {}
        for nd in plan["placement"]:
            for i in (nd.get("indexes") or []):
                byname.setdefault(i["name"], []).append(str(i["instId"]))
        env = {a: b for a, b in os.environ.items() if not a.startswith(("CBAUTH", "NS_SERVER"))}
        work = tempfile.mkdtemp(prefix="qip_fresh_")
        rr, op = os.path.join(work, "rr.json"), os.path.join(work, "out.json")

        def read_out():
            try:
                pj = json.load(open(op))
            except Exception:
                return None
            pl, pool = {}, {n: list(v) for n, v in byname.items()}
            for nd in pj["placement"]:
                for i in (nd.get("indexes") or []):
                    reps = pool.get(i["name"])
                    if not reps or nd["nodeId"] not in nodes_l:
                        return None
                    pl[reps.pop()] = nd["nodeId"]
            return pl if len(pl) == len(anchor) else None

        # planner: -command=rebalance starting from the round-robin plan, restarted until the budget ends
        cx.write_plan(rr, plan)
        args = [BIN, "-command=rebalance", f"-plan={rr}", f"-output={op}",
                "-enableShardAffinity=false"]
        subprocess.run(args, capture_output=True, timeout=120, env=env)   # warm-up, untimed
        tp = {"pts": [], "next": 0}
        best, prev, runs = float("inf"), float("inf"), 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < budget:
            prev = best
            if os.path.exists(op):
                os.remove(op)
            try:
                subprocess.run(args, capture_output=True, timeout=120, env=env)
            except Exception:
                pass
            runs += 1
            snap(tp, t0, (prev, float("nan"), 0.0), budget)
            pl = read_out()
            if pl:
                best = min(best, score(pl))
        snap(tp, t0, (prev, float("nan"), 0.0), budget)
        out["planner"] = dict(pts=tp["pts"], final=best, runs=runs)

    # ---- QUBO: no movement term; lambda = B+1, beta = 5B, kappa = 1, nu = 20
    B = BF.penalty_weight_bound(inst)
    q = build_move_qubo(inst, anchor, lam=B + 1.0, kappa=1.0, beta=AT.BHAT * B,
                        capacity_mode="unbalanced", onedest_mult=20.0,
                        res_weights={"d": 3.0, "m": 0.0, "r": 1.0})[0]
    vl, M, off = S.to_dense(q)
    case = dict(inst=inst, P0=anchor, data_by=res["data"], mem_by=res["mem"], q=q,
                eject=None, names=nodes_l)
    tq = {"pts": [], "next": 0}
    bq, prev, nt, nf, b = float("inf"), float("inf"), 0, 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget:
        prev = bq
        reads = min(20, 2 ** b) if b < 5 else 20
        xs = S.s_neal(vl, M, off, q, case, reads=reads, sweeps=400, seed=b)[0]
        snap(tq, t0, (prev, float("nan"), 0.0), budget)
        for x in xs:
            nt += 1
            pl, conf = decode_move({v: int(x[i]) for i, v in enumerate(vl)}, inst, anchor)
            if conf or not place_feasible(pl, inst):
                continue                                   # discarded, not repaired
            nf += 1
            bq = min(bq, score(pl))
        b += 1
    snap(tq, t0, (prev, float("nan"), 0.0), budget)
    out["qubo"] = dict(pts=tq["pts"], final=bq, reads=nt, feas=nf, batches=b, nvars=len(vl))
    return out


def _rebalance_setup(scen, util, k, perf, seed):
    import exp_cbcost as E, cb_scenarios as CS
    from cb_instance import instance_multi
    from cb_cost import cb_cost, new_nodes_of
    plan, eject = CS.make_scenario(scen, k, seed, None, perf=perf, util=util)
    inst, P0, res_by, _ = instance_multi(plan, eject=eject)
    nodes = [n.name for n in inst.usable_nodes()]
    allnodes = [n["nodeId"] for n in plan["placement"]]
    newn = new_nodes_of(plan)
    gone = {eject} if eject else ()

    def score(pl):
        c, parts = cb_cost(pl, allnodes, res_by, P0, newn, gone)
        return c, parts["cvs"].get("data", float("nan")), parts["data_moved_pct"]
    E.set_scorer(score)
    return E, plan, eject, inst, P0, res_by, nodes, newn


def rebalance_cell(scen, util, k, perf, seed, budget=300.0, qubo_only=False):
    E, plan, eject, inst, P0, res_by, nodes, newn = _rebalance_setup(scen, util, k, perf, seed)
    data_by, mem_by = res_by["data"], res_by["mem"]
    tot = sum(data_by.values())
    c0, U0, _ = E.cb_parts(P0, nodes, data_by, P0, tot)
    out = dict(experiment="couchbase_rebalance", scen=scen, util=util, k=k, perf=perf,
               seed=seed, budget=budget, cost0=c0, U0=U0, new_nodes=sorted(newn),
               repair=False)
    tag = f"{scen}_{k}_{perf}_{int(round(util * 1000))}_{seed}"
    if not qubo_only:
        _planner_bin()
        pp, pm = E.run_planner(plan, nodes, data_by, P0, tot, budget, tag, eject=eject)
        out["planner"] = dict(pts=pp, **pm)
    qp, qm = E.run_qubo(inst, P0, data_by, mem_by, nodes, tot, budget)
    out["qubo"] = dict(pts=qp, **qm)
    return out


LEVELS = [0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25]


def budget_cell(scen, util, k, perf, seed, budget=300.0):
    import anytime as AT, bench_full as BF, solvers_bench as S
    from qubo_move_variables import build_move_qubo, decode_move
    E, plan, eject, inst, P0, res_by, nodes, newn = _rebalance_setup(scen, util, k, perf, seed)
    data_by, mem_by = res_by["data"], res_by["mem"]
    tot = sum(data_by.values()); quota = plan["memQuota"]
    allnodes = [n["nodeId"] for n in plan["placement"]]

    def legal(pl):
        load = {n: 0.0 for n in allnodes}
        for r, n in pl.items():
            load[n] += res_by["mem"][r]
        return max(load.values()) <= quota * (1 + 1e-9)

    B = BF.penalty_weight_bound(inst)
    dmin = max(1.0, min(v for v in data_by.values() if v > 0))
    w = {kk: v / dmin for kk, v in data_by.items()}
    share = budget / len(LEVELS)
    front, rows, best = [], [], [float("inf"), float("nan"), float("nan")]
    t0 = time.perf_counter()
    for i, mb in enumerate(LEVELS):
        deadline = t0 + share * (i + 1)
        if time.perf_counter() >= deadline:
            continue
        q = build_move_qubo(inst, P0, lam=B + 1.0, w_move=w, move_budget=mb,
                            capacity_mode="unbalanced", kappa=1.0, beta=AT.BHAT * B,
                            move_quad=False, new_nodes=newn, onedest_mult=20.0,
                            res_weights={"d": 3.0, "m": 1.0, "r": 1.0})[0]
        case = dict(inst=inst, P0=P0, data_by=data_by, mem_by=mem_by, q=q, eject=None, names=nodes)
        vl, M, off = S.to_dense(q)
        b, nt, nleg = 0, 0, 0
        while time.perf_counter() < deadline:
            for x in S.s_neal(vl, M, off, q, case, reads=200, sweeps=400, seed=b)[0]:
                nt += 1
                pl, conf = decode_move({v: int(x[j]) for j, v in enumerate(vl)}, inst, P0)
                if conf or not S.evaluate(x, vl, case, M, off)["feasible"] or not legal(pl):
                    continue                                   # discarded, not repaired
                nleg += 1
                c, U, mv = E.cb_parts(pl, nodes, data_by, P0, tot)
                E.pareto_add(front, mv, U)
                if c < best[0]:
                    best = [c, U, mv]
            b += 1
        rows.append(dict(budget=mb, samples=nt, legal=100.0 * nleg / max(nt, 1)))
    return dict(experiment="couchbase_budget", scen=scen, util=util, k=k, perf=perf,
                seed=seed, budget=budget,
                budgeted=dict(levels=rows, front=sorted(front), cost=best[0],
                              balance=best[1], moved=best[2]))
