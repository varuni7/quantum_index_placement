#!/usr/bin/env python3
"""Couchbase instance helpers: TPC-H size reference, instance construction, placement checks."""
import os
import sys
import json
import math
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators import cbindexplan_io as cx
from paper_qubo_verify import Replica, Node, Instance, penalty_weight_bound  # noqa: F401 (used as bench_full.penalty_weight_bound)

import tempfile
TMP = os.environ.get("QIP_TMP", os.path.join(tempfile.gettempdir(), "qip"))
os.makedirs(TMP, exist_ok=True)
K_IDX, COPIES = 30, 3


def sizes_for(seed, skew, k=K_IDX):
    if skew is None:
        real = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "external", "hyrise-tpch-sf10", "index_sizes.json")
        rows = [r for r in json.load(open(real))
                if r["columns"][0].split("_")[0] not in ("n", "r")]
        return [max(1, round(p["size_bytes"] / cx.BYTES_PER_DOC))
                for p in random.Random(seed).sample(rows, k)]
    raise ValueError("sizes_for supports only skew=None (TPC-H sizes)")


def instance_from(plan, eject=None):
    rows, P0 = [], {}
    for nd in plan["placement"]:
        for i in (nd.get("indexes") or []):
            data, mem = cx.moi_mem_total(i["numOfDocs"])
            rows.append((str(i["instId"]), i["defnId"], data, mem))
            P0[str(i["instId"])] = nd["nodeId"]
    scale = max(1.0, min(m for _, _, _, m in rows))
    reps = [Replica(iid, f"L{d}", m / scale, m / scale, m / scale, True)
            for iid, d, _, m in rows]
    cap = plan["memQuota"] / scale
    nodes = [Node(nd["nodeId"], cap, cap, cap, group="g0",
                  excluded=(eject is not None and nd["nodeId"] == eject))
             for nd in plan["placement"]]
    return (Instance(reps, nodes, enforce_group_cover=False), P0,
            {iid: dt for iid, _, dt, _ in rows}, {iid: m for iid, _, _, m in rows})


def cv_of(place, names, mem_by):
    load = {n: 0.0 for n in names}
    for iid, nd in place.items():
        if nd in load:
            load[nd] += mem_by[iid]
    v = list(load.values())
    if not v:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) / m if m else 0.0


def audit(place, inst, P0, eject=None):
    if len(place) < len(P0):
        return False
    if eject and any(n == eject for n in place.values()):
        return False
    bylog = {}
    for r in inst.replicas:
        bylog.setdefault(r.logical, []).append(place.get(r.name))
    return all(len(set(v)) == len(v) for v in bylog.values())


def stats(P0, P1, data_by):
    mv = [i for i in P0 if P1.get(i) != P0[i]]
    tot = sum(data_by.values())
    return (100 * sum(data_by[i] for i in mv) / tot, sum(data_by[i] for i in mv) / 1e9)


