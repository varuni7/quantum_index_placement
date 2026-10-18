#!/usr/bin/env python3
"""QUBO container."""

from dataclasses import dataclass, field
from collections import defaultdict




@dataclass
class Index:
    name: str
    copies: int
    resources: dict
    equiv_group: str = None

    def __post_init__(self):
        if self.equiv_group is None:
            self.equiv_group = self.name


@dataclass
class Cluster:
    nodes: list
    mem_quota: dict
    server_groups: dict = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    deleted: set = field(default_factory=set)

    def usable_nodes(self):
        return [n for n in self.nodes if n not in self.excluded and n not in self.deleted]


@dataclass
class Copy:
    cid: int
    index_name: str
    equiv_group: str
    resources: dict


class QUBO:
    def __init__(self):
        self.Q = defaultdict(float)
        self.offset = 0.0

    def add_linear(self, v, bias):
        key = (v, v)
        self.Q[key] += bias

    def add_quadratic(self, u, v, bias):
        if u == v:
            self.add_linear(u, bias)
            return
        key = (u, v) if u < v else (v, u)
        self.Q[key] += bias

    def add_offset(self, c):
        self.offset += c

    def energy(self, sample):
        e = self.offset
        for (u, v), bias in self.Q.items():
            if u == v:
                e += bias * sample.get(u, 0)
            else:
                e += bias * sample.get(u, 0) * sample.get(v, 0)
        return e

    def variables(self):
        vs = set()
        for (u, v) in self.Q:
            vs.add(u)
            vs.add(v)
        return sorted(vs)
