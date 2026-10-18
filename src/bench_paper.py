#!/usr/bin/env python3
"""Elasticsearch instances: index sizes and the order of per-index quantities."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as AT, bench_full as BF
import generators.cbindexplan_io as cx


def instance(k, seed, skew):
    plan = AT.make_plan(k, seed, float(skew))
    inst, P0, data_by, mem_by = BF.instance_from(plan)
    docs, seen = [], set()
    for nd in plan["placement"]:
        for i in nd["indexes"]:
            if i["name"] not in seen:
                seen.add(i["name"]); docs.append(i["numOfDocs"])
    szs = [int(cx.moi_mem_total(d)[0]) for d in docs]
    unit = sum(data_by.values()) / max(len(data_by), 1)
    return plan, inst, P0, data_by, szs, unit


def logical_order(plan):
    seen, order = set(), []
    for nd in plan["placement"]:
        for i in nd["indexes"]:
            if i["name"] not in seen:
                seen.add(i["name"]); order.append(f"L{i['defnId']}")
    return order


def es_row(inst):
    seen, row = set(), {}
    for r in inst.replicas:
        L = str(r.logical)
        if L not in seen:
            seen.add(L); row[L] = len(row)
    return row
