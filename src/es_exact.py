#!/usr/bin/env python3
"""Phi, the objective Elasticsearch's balancer decreases (paper, Eq. 20)."""
import os
import sys
import numpy as np
np.infty = getattr(np, "infty", np.inf)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_qubo import TH0, TH1, TH2, TH3


def phi_of_placement(pl, inst, data_by, unit, wload_by=None):
    nodes = [n.name for n in inst.usable_nodes()]
    N = len(nodes)
    logicals = sorted({r.logical for r in inst.replicas})
    S = {n: 0 for n in nodes}; D = {n: 0.0 for n in nodes}; W = {n: 0.0 for n in nodes}
    SI = {(L, n): 0 for L in logicals for n in nodes}
    use_wl = wload_by is not None and any(wload_by.values())
    for r in inst.replicas:
        n = pl[r.name]
        S[n] += 1; D[n] += data_by[r.name] / unit; SI[(r.logical, n)] += 1
        if use_wl: W[n] += wload_by[r.name]
    avgS = len(inst.replicas) / N
    avgD = sum(D.values()) / N
    avgW = sum(W.values()) / N
    tot = 0.0
    th3 = TH3 * unit
    for L in logicals:
        avgI = sum(1 for r in inst.replicas if r.logical == L) / N
        for n in nodes:
            w = (TH0*(S[n]-avgS) + TH1*(SI[(L, n)]-avgI)
                 + TH2*(W[n]-avgW) + th3*(D[n]-avgD))
            tot += w*w
    return tot
