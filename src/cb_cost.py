#!/usr/bin/env python3
"""The Couchbase planner's objective (paper, Eq. 19), used to score both methods."""
import math

RES = ("mem", "data", "disk", "scan", "drain")


def node_loads(pl, nodes, by):
    on = {n: 0.0 for n in nodes}
    for rep, nd in pl.items():
        if nd in on:
            on[nd] += by[rep]
    return [on[n] for n in nodes]


def _cv(vals):
    m = sum(vals) / len(vals)
    if m == 0:
        return None
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / m


def cb_cost(pl, nodes, res_by, P0, new_nodes=(), ejected=(), data_cost_weight=1.0,
            mem_cost_weight=1.0):
    live = [n for n in nodes
            if not (n in set(ejected) and not any(v == n for v in pl.values()))]
    cvs = {}
    for r in RES:
        if r in res_by:
            c = _cv(node_loads(pl, live, res_by[r]))
            if c is not None:
                cvs[r] = c

    mem_cost = cvs.get("mem", 0.0) * mem_cost_weight
    data_cost = cvs.get("data", 0.0)
    count = 2 + sum(1 for r in ("disk", "scan", "drain") if r in cvs)
    avg_resource = (mem_cost + data_cost + cvs.get("disk", 0.0)
                    + cvs.get("drain", 0.0) + cvs.get("scan", 0.0)) / count

    usage_cost = (mem_cost + data_cost) / 2.0
    w = data_cost_weight * (1.0 - usage_cost)

    new, gone = set(new_nodes), set(ejected)
    keep = [r for r in P0 if pl.get(r) not in new and P0[r] not in gone]
    tot_data = sum(res_by["data"][r] for r in keep)
    tot_idx = len(keep)
    d_moved = sum(res_by["data"][r] for r in keep if pl.get(r) != P0[r])
    i_moved = sum(1 for r in keep if pl.get(r) != P0[r])
    move_cost = (d_moved / tot_data * w) if tot_data else 0.0
    index_cost = (i_moved / tot_idx * w) if tot_idx else 0.0
    avg_move = (index_cost + move_cost) / 2.0

    cost = (avg_resource + avg_move) / 2.0
    return cost, dict(cvs=cvs, count=count, avg_resource=avg_resource,
                      usage_cost=usage_cost, data_moved_pct=100.0 * d_moved / tot_data
                      if tot_data else 0.0,
                      index_moved_pct=100.0 * i_moved / tot_idx if tot_idx else 0.0,
                      counted_pct=100.0 * len(keep) / max(len(P0), 1))


def new_nodes_of(plan):
    return {nd["nodeId"] for nd in plan["placement"] if not (nd.get("indexes") or [])}
