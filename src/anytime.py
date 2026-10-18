#!/usr/bin/env python3
"""Synthetic Couchbase instance generator (index sizes, scan and drain rates, plans)."""
import os
import sys
import math
import random
from collections import defaultdict
import numpy as np
np.infty = getattr(np, "infty", np.inf)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_full as BF
import generators.cbindexplan_io as cx

NBASE, NEXTRA, COPIES = 5, 1, 2
LIVE = os.environ.get("LIVE", "1") != "0"
RATE_SKEW = float(os.environ.get("RATE_SKEW", 50.0))
CAP_UTIL = float(os.environ.get("CAP_UTIL", 2.0 / 3.0))
BHAT = 5.0
CHECKPOINTS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]


def _raw_sizes(kidx, seed, skew):
    if skew is None:
        docs = list(BF.sizes_for(seed, skew, k=kidx))[:kidx]
        while len(docs) < kidx:
            docs += list(BF.sizes_for(seed + len(docs), skew, k=kidx))[:kidx - len(docs)]
        return docs
    if skew <= 1:
        return [1_000_000] * kidx
    rng = random.Random(seed * 7919 + kidx)
    d = sorted(math.exp(rng.uniform(0.0, math.log(float(skew)))) for _ in range(kidx))
    d[0], d[-1] = 1.0, float(skew)
    return [max(1, round(x * 1_000_000)) for x in d]


def _rates(kidx, seed):
    rng = random.Random(seed * 104729 + kidx * 31 + 7)
    scan = [math.exp(rng.uniform(0.0, math.log(RATE_SKEW))) for _ in range(kidx)]
    drain = [math.exp(rng.uniform(0.0, math.log(RATE_SKEW))) for _ in range(kidx)]
    return scan, drain


def make_plan(kidx, seed, skew):
    docs = _raw_sizes(kidx, seed, skew)
    if skew is not None:
        tgt = sum(_raw_sizes(kidx, seed, None))
        cur = sum(docs) or 1
        docs = [max(1, round(d * tgt / cur)) for d in docs]
    mem_each = [cx.moi_mem_total(x)[1] for x in docs]
    nodes_total = NBASE + NEXTRA
    quota = max(1.5 * max(mem_each), COPIES * sum(mem_each) / (nodes_total * CAP_UTIL))
    nodes = [{"nodeId": f"node{j}", "serverGroup": "g0", "indexes": []}
             for j in range(NBASE + NEXTRA)]
    scan, drain = _rates(kidx, seed)
    iid = 1000
    for L, nd in enumerate(docs):
        for c in range(COPIES):
            ix = cx._index_instance(L + 1, iid, f"idx{L}", nd)
            if LIVE:
                ix.update(cx.live_fields(nd, mem_each[L], scan[L], drain[L]))
            nodes[(L + c) % NBASE]["indexes"].append(ix)
            iid += 1
    plan = {"memQuota": int(quota), "cpuQuota": 256, "placement": nodes,
            "_util": COPIES * sum(mem_each) / (nodes_total * quota)}
    if LIVE:
        plan["isLive"] = True
        for nd in nodes:
            agg = lambda f: int(sum(ix.get(f, 0) for ix in nd["indexes"]))
            nd.update({"actualMemUsage": agg("actualMemUsage"),
                       "actualMemOverhead": 0,
                       "actualMemMin": agg("actualMemMin"),
                       "actualDataSize": agg("actualDataSize"),
                       "actualDiskUsage": agg("actualDiskUsage"),
                       "actualScanRate": agg("actualScanRate"),
                       "actualDrainRate": agg("actualDrainRate")})
    return plan


def roundrobin(inst):
    nodes = [n.name for n in inst.usable_nodes()]
    bylog = defaultdict(list)
    for r in inst.replicas:
        bylog[r.logical].append(r.name)
    pl = {}
    for li, (_lg, reps) in enumerate(sorted(bylog.items())):
        for c, rep in enumerate(reps):
            pl[rep] = nodes[(li + c) % len(nodes)]
    return pl
