#!/usr/bin/env python3
"""neal sampling helpers."""

from collections import defaultdict


def _add_squared(q, terms, const, w):
    c = defaultdict(float)
    for v, coef in terms:
        c[v] += coef
    items = list(c.items())
    q.add_offset(w * const * const)
    for i, (v, cv) in enumerate(items):
        q.add_linear(v, w * (cv * cv + 2.0 * const * cv))
        for (v2, cv2) in items[i + 1:]:
            q.add_quadratic(v, v2, w * 2.0 * cv * cv2)


def qubo_to_bqm(q):
    import dimod
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    for (u, v), bias in q.Q.items():
        if u == v:
            bqm.add_linear(u, bias)
        else:
            bqm.add_quadratic(u, v, bias)
    bqm.offset = q.offset
    return bqm


def sample_with_neal(q, reads, sweeps, seed=None):
    import neal
    bqm = qubo_to_bqm(q)
    sampler = neal.SimulatedAnnealingSampler()
    kw = {} if seed is None else {"seed": seed}
    ss = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps, **kw)
    return [dict(datum.sample) for datum in ss.data(["sample"])]
