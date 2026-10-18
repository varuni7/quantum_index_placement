#!/usr/bin/env python3
"""Anti-affinity check for Elasticsearch placements: no two copies of an index on one node."""
from collections import defaultdict


def same_shard_ok(pl, inst):
    seen = defaultdict(set)
    for r in inst.replicas:
        n = pl.get(r.name)
        if n is None:
            return False
        if r.logical in seen[n]:
            return False
        seen[n].add(r.logical)
    return True
