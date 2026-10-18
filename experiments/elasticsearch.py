"""Elasticsearch experiments (Section 7): our QUBO and Elasticsearch's allocator, both scored by Phi."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import numpy as np
np.infty = getattr(np, "infty", np.inf)


def qubo_cell(k, backing, seed, nodes=8, reads=3000, lamx=20.0, lamdiv=3000.0):
    import anytime as AT
    AT.NBASE, AT.NEXTRA = nodes - 1, 1
    import bench_paper as BP, paper_qubo_experiment as E, neal
    from es_qubo import build_es_qubo
    from es_exact import phi_of_placement
    from qubo_move_variables import decode_move
    from write_load import elastic_logs_loads
    from es_cap import driver_cap
    from es_legal import same_shard_ok
    t0 = time.perf_counter()
    plan, inst, P0, data_by, szs, unit = BP.instance(k, seed, 100.0)
    logicals = BP.logical_order(plan)
    per = elastic_logs_loads(len(logicals), seed, n_streams=max(1, k // backing))
    wmap = dict(zip(logicals, per))
    wload_by = {r.name: wmap[r.logical] for r in inst.replicas}
    cap = driver_cap(inst, P0, data_by, nodes)
    anchor = AT.roundrobin(inst)
    PHI = lambda pl: phi_of_placement(pl, inst, data_by, unit, wload_by=wload_by)
    q, meta = build_es_qubo(inst, anchor, data_by, wload_by=wload_by, cap_per_node=cap,
                            move_budget=0.0, factored=True)
    q, meta = build_es_qubo(inst, anchor, data_by, wload_by=wload_by, cap_per_node=cap,
                            move_budget=0.0, factored=True, onedest_mult=lamx,
                            lam=meta["lam"] / lamdiv)
    bqm = E.qubo_to_bqm(q)
    sampler = neal.SimulatedAnnealingSampler()
    best, nfeas, i = float("inf"), 0, 0
    while i * 20 < reads:
        for d in sampler.sample(bqm, num_reads=20, num_sweeps=400, seed=i).data(["sample"]):
            pl, conf = decode_move(dict(d.sample), inst, anchor)
            if conf or not same_shard_ok(pl, inst):
                continue                                   # discarded, not repaired
            nfeas += 1
            best = min(best, PHI(pl))
        i += 1
    return dict(experiment="es_qubo", k=k, backing=backing, seed=seed, nodes=nodes,
                reads=reads, nvars=len(bqm.variables), phi_anchor=PHI(anchor),
                phi=None if best == float("inf") else best, nfeas=nfeas,
                secs=time.perf_counter() - t0)


def allocator_cell(k, backing, seed, nodes=8, restarts=2):
    if not os.environ.get("ES_REPO"):
        raise SystemExit("Set ES_REPO to an elasticsearch v9.5.3 checkout with the harness "
                         "applied (elasticsearch_harness/README.md).")
    os.environ["SW_NODES"] = str(nodes); os.environ["SW_NREST"] = str(restarts)
    os.environ.setdefault("SW_OUT", os.path.join(os.getcwd(), "es_allocator_out"))
    import es_sweep_es as ES
    r = ES.cell(k, backing, seed); r["experiment"] = "es_allocator"
    return r
