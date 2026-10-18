#!/usr/bin/env python3
"""Rebalancing scenarios: quota down, quota up, node added."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as AT
import generators.cbindexplan_io as cx

RESIDENT_RATIO = 50


def _plan(kidx, seed, skew, n_pop, n_empty, quota_mult=1.0, perf=None,
          util=None, light_frac=None, light_every=4):
    seckeys = None
    if perf:
        import cb_perfrunner as PR
        rows = PR.perfrunner_sizes(kidx, seed, ratio_config=perf,
                                   resident_ratio=RESIDENT_RATIO)
        docs = [max(1, round(d)) for _, _, d, _ in rows]
        mem_each = [m for _, _, _, m in rows]
        seckeys = [int(round(ks)) for _, ks, _, _ in rows]
    else:
        docs = AT._raw_sizes(kidx, seed, skew)
        if skew is not None:
            tgt = sum(AT._raw_sizes(kidx, seed, None))
            cur = sum(docs) or 1
            docs = [max(1, round(d * tgt / cur)) for d in docs]
        mem_each = [cx.moi_mem_total(x)[1] for x in docs]
    total_nodes = n_pop + n_empty
    UTIL = 0.80
    if util is not None:
        quota = max(max(mem_each),
                    quota_mult * AT.COPIES * sum(mem_each) / (util * total_nodes))
    else:
        quota = quota_mult * max(max(mem_each),
                                 AT.COPIES * sum(mem_each) / (UTIL * total_nodes))
    nodes = [{"nodeId": f"node{j}", "serverGroup": "g0", "indexes": []}
             for j in range(total_nodes)]
    scan, drain = AT._rates(kidx, seed)
    iid = 1000
    for L, nd in enumerate(docs):
        for c in range(AT.COPIES):
            ix = cx._index_instance(L + 1, iid, f"idx{L}", nd)
            if seckeys:
                ix["secKeySize"] = seckeys[L]
            ix.update(cx.live_fields(nd, mem_each[L], scan[L], drain[L]))
            tgt = (L + c) % n_pop
            if light_frac is not None and tgt == n_pop - 1 and (L % light_every):
                tgt = (L + c) % (n_pop - 1)
            nodes[tgt]["indexes"].append(ix)
            iid += 1
    plan = {"memQuota": int(quota), "cpuQuota": 256, "placement": nodes, "isLive": True}
    for nd in nodes:
        agg = lambda f: int(sum(i.get(f, 0) for i in nd["indexes"]))
        nd.update({"actualMemUsage": agg("actualMemUsage"), "actualMemOverhead": 0,
                   "actualMemMin": agg("actualMemMin"),
                   "actualDataSize": agg("actualDataSize"),
                   "actualDiskUsage": agg("actualDiskUsage"),
                   "actualScanRate": agg("actualScanRate"),
                   "actualDrainRate": agg("actualDrainRate")})
    return plan


NB = AT.NBASE


def make_scenario(scen, kidx, seed, skew, n_nodes=None, perf=None, util=None):
    nb = (n_nodes - 1) if n_nodes else NB
    if scen == "add-node-1":
        return _plan(kidx, seed, skew, nb, 1, perf=perf, util=util), None
    if scen == "add-node-2":
        return _plan(kidx, seed, skew, nb, 2, perf=perf, util=util), None
    if scen == "eject-node":
        return _plan(kidx, seed, skew, nb + 1, 0, perf=perf, util=util), f"node{nb}"
    if scen == "quota-up":
        return _plan(kidx, seed, skew, nb + 1, 0, quota_mult=1.5, perf=perf, util=util), None
    if scen == "scale-out":
        return _plan(kidx, seed, skew, nb + 1, 0, perf=perf, util=util,
                     light_frac=True), None
    if scen == "quota-down":
        return _plan(kidx, seed, skew, nb + 1, 0, quota_mult=0.85, perf=perf, util=util), None
    raise ValueError(scen)


