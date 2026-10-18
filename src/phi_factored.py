#!/usr/bin/env python3
"""Checks the factored form of Phi against direct summation, coefficient by coefficient.
usage: python src/phi_factored.py"""
import os
import sys
import numpy as np
np.infty = getattr(np, "infty", np.inf)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as AT
from es_qubo import build_es_qubo


def compare(q1, q2, tol=1e-6):
    keys = set(q1.Q) | set(q2.Q)
    worst, nbad = 0.0, 0
    for k in keys:
        a, b = q1.Q.get(k, 0.0), q2.Q.get(k, 0.0)
        scale = max(abs(a), abs(b), 1e-12)
        rel = abs(a - b) / scale
        if rel > tol and abs(a - b) > 1e-9:
            nbad += 1
        worst = max(worst, rel if abs(a - b) > 1e-9 else 0.0)
    return nbad, worst, len(keys)


if __name__ == "__main__":
    AT.NBASE, AT.NEXTRA = 7, 1          # eight nodes, as in the Elasticsearch experiments
    import bench_paper as BP
    from write_load import elastic_logs_loads
    print(f"  {'k':>5} {'b':>3} {'mismatched coefficients':>24} {'worst rel. error':>17}")
    for k in [int(x) for x in os.environ.get("SIZES", "16,32,64,96,128").split(",")]:
        for b in (2, 8):
            plan, inst, P0, data_by, szs, unit = BP.instance(k, 9000, 100.0)
            per = elastic_logs_loads(len(szs), 9000, n_streams=max(1, k // b))
            wmap = dict(zip(BP.logical_order(plan), per))
            wl = {r.name: wmap[r.logical] for r in inst.replicas}
            qf, _ = build_es_qubo(inst, P0, data_by, wload_by=wl, lam=0.0, factored=True)
            qd, _ = build_es_qubo(inst, P0, data_by, wload_by=wl, lam=0.0, factored=False)
            nbad, worst, tot = compare(qd, qf)
            print(f"  {k:>5} {b:>3} {str(nbad) + ' / ' + str(tot):>24} {worst:>17.2e}", flush=True)
