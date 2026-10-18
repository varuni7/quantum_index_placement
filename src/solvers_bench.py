#!/usr/bin/env python3
"""Dense QUBO conversion, neal sampling, and decoding with legality checks."""
import os
import sys
import time
import numpy as np

np.infty = getattr(np, "infty", np.inf)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_full import audit, cv_of, stats
from qubo_move_variables import decode_move
from cqm_experiment import place_feasible



def to_dense(q):
    varlist = sorted(q.variables())
    idx = {v: i for i, v in enumerate(varlist)}
    M = np.zeros((len(varlist), len(varlist)), dtype=np.float64)
    for (u, v), bias in q.Q.items():
        i, j = idx[u], idx[v]
        if i == j:
            M[i, i] += bias
        else:
            a, b = (i, j) if i < j else (j, i)
            M[a, b] += bias
    return varlist, M, q.offset


def energy(x, M, offset):
    return float(x @ M @ x) + offset


def evaluate(xvec, varlist, case, M, offset):
    sample = {v: int(xvec[i]) for i, v in enumerate(varlist)}
    pl, conf = decode_move(sample, case["inst"], case["P0"])
    e = energy(np.asarray(xvec, dtype=np.float64), M, offset)
    ok = (conf == 0 and place_feasible(pl, case["inst"])
          and audit(pl, case["inst"], case["P0"], case["eject"]))
    if not ok:
        return dict(energy=e, feasible=False, cv=float("nan"), dp=float("nan"))
    dp, gb = stats(case["P0"], pl, case["data_by"])
    return dict(energy=e, feasible=True,
                cv=cv_of(pl, case["names"], case["mem_by"]), dp=dp, gb=gb)


def s_neal(varlist, M, offset, q, case, reads=2500, sweeps=400, seed=0):
    import paper_qubo_experiment as E
    t = time.perf_counter()
    samples = E.sample_with_neal(q, reads, sweeps, seed=seed)
    dt = time.perf_counter() - t
    idx = {v: i for i, v in enumerate(varlist)}
    out = np.zeros((len(samples), len(varlist)), dtype=np.float64)
    for r, s in enumerate(samples):
        for v, bit in s.items():
            out[r, idx[v]] = bit
    return list(out), dt, f"{reads} reads x {sweeps} sweeps"
