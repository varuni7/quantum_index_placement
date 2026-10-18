#!/usr/bin/env python3
"""Couchbase instance for the QUBO: bytes, scan and drain as the three resources."""
from paper_qubo_verify import Replica, Node, Instance


def instance_multi(plan, eject=None):
    rows, P0 = [], {}
    for nd in plan["placement"]:
        for ix in (nd.get("indexes") or []):
            iid = str(ix["instId"])
            rows.append((iid, ix["defnId"],
                         float(ix["actualDataSize"]),
                         float(ix["actualScanRate"]),
                         float(ix["actualDrainRate"])))
            P0[iid] = nd["nodeId"]

    scale = max(1.0, min(b for _, _, b, _, _ in rows))
    reps = [Replica(iid, f"L{d}", b / scale, s, dr, True)
            for iid, d, b, s, dr in rows]
    cap = plan["memQuota"] / scale
    big_s = 1e6 * max(sum(r.m for r in reps), 1.0)
    big_r = 1e6 * max(sum(r.r for r in reps), 1.0)
    nodes = [Node(nd["nodeId"], cap, big_s, big_r, group="g0",
                  excluded=(eject is not None and nd["nodeId"] == eject))
             for nd in plan["placement"]]
    inst = Instance(reps, nodes, enforce_group_cover=False)

    by = {"mem": {}, "data": {}, "disk": {}, "scan": {}, "drain": {}}
    for iid, _d, b, s, dr in rows:
        by["mem"][iid] = b
        by["data"][iid] = b
        by["disk"][iid] = b
        by["scan"][iid] = s
        by["drain"][iid] = dr
    return inst, P0, by, scale
