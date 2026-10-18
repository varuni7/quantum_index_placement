#!/usr/bin/env python3
"""Placement feasibility checks shared by the experiments."""

from collections import defaultdict

from paper_qubo_verify import RESOURCES, footprint, capacity


def place_feasible(place, inst):
    nodes = inst.usable_nodes()
    reps = inst.replicas
    for rep in reps:
        if place.get(rep.name) is None:
            return False
    by = defaultdict(list)
    for rep in reps:
        by[rep.logical].append(rep)
    for grp in by.values():
        seen = set()
        for r in grp:
            if place[r.name] in seen:
                return False
            seen.add(place[r.name])
    for g, gn in inst.groups().items():
        gs = {n.name for n in gn}
        if not any(place[rep.name] in gs for rep in reps):
            return False
    for n in nodes:
        for res in RESOURCES:
            use = sum(footprint(rep, res) for rep in reps if place[rep.name] == n.name)
            if use > capacity(n, res) + 1e-9:
                return False
    return True
