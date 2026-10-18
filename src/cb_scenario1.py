#!/usr/bin/env python3
"""Fresh placement instances, anchored at a size-blind round robin."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import anytime as AT
import cb_scenarios as CS
from cb_instance import instance_multi


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


def make_s1(kidx, seed, skew, n_nodes=None, perf=None, util=None):
    n_nodes = n_nodes or AT.NBASE + 1
    plan, _ = CS.make_scenario("quota-up", kidx, seed, skew, n_nodes=n_nodes,
                               perf=perf, util=util)
    inst, _P0, res, _ = instance_multi(plan)
    anchor = roundrobin(inst)
    nodes = [n.name for n in inst.usable_nodes()]
    return inst, anchor, res, nodes, plan
