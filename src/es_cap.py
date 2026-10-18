"""Per-node disk capacity, identical to the one the Elasticsearch harness sets."""
def driver_cap(inst, P0, data_by, nodes_n):
    used = {}
    for r in inst.replicas:
        used[P0[r.name]] = used.get(P0[r.name], 0.0) + data_by[r.name]
    return max(sum(data_by.values()) / nodes_n, max(used.values())) * 2.0
