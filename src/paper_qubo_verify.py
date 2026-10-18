#!/usr/bin/env python3
"""Problem model (replicas, nodes, instances) and the penalty-weight bound B."""

from dataclasses import dataclass

RESOURCES = ("d", "m", "r")


@dataclass
class Replica:
    name: str
    logical: str
    d: float
    m: float
    r: float
    mandatory: bool = True


@dataclass
class Node:
    name: str
    D: float
    M: float
    R: float
    group: str = "g0"
    excluded: bool = False


@dataclass
class Instance:
    replicas: list
    nodes: list
    alpha_d: float = 1.0
    alpha_m: float = 1.0
    alpha_r: float = 1.0
    beta_g: float = 1.0
    enforce_group_cover: bool = True

    def usable_nodes(self):
        return [n for n in self.nodes if not n.excluded]

    def groups(self):
        gs = {}
        for n in self.usable_nodes():
            gs.setdefault(n.group, []).append(n)
        return gs


def footprint(rep, res):
    return {"d": rep.d, "m": rep.m, "r": rep.r}[res]


def capacity(node, res):
    return {"d": node.D, "m": node.M, "r": node.R}[res]


def penalty_weight_bound(inst):
    reps = inst.replicas
    N = len(inst.usable_nodes())
    G = len(inst.groups())
    Dtot = sum(rep.d for rep in reps)
    Mtot = sum(rep.m for rep in reps)
    Rtot = sum(rep.r for rep in reps)
    Itot = len(reps)
    B = (inst.alpha_d * N * Dtot ** 2 + inst.alpha_m * N * Mtot ** 2
         + inst.alpha_r * N * Rtot ** 2 + inst.beta_g * G * Itot ** 2)
    return B
