#!/usr/bin/env python3
"""Runs Elasticsearch's allocator on one instance through the harness (needs ES_REPO)."""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.infty = getattr(np, "infty", np.inf)
from collections import defaultdict
import anytime as AT

NODES = int(os.environ.get("SW_NODES", "8"))
AT.NBASE, AT.NEXTRA = NODES - 1, 1

import bench_paper as BP
from es_exact import phi_of_placement
from write_load import elastic_logs_loads
from es_cap import driver_cap

import tempfile
ES_REPO = os.environ.get("ES_REPO", "")
ES_GRADLE = os.environ.get("ES_GRADLE", os.path.join(ES_REPO, "gradlew"))
WORK = os.path.join(tempfile.gettempdir(), os.environ.get("SW_OUT", "es_sweep_es"))
OUT = os.environ.get("SW_OUT", "es_sweep_es")
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
NREST = int(os.environ.get("SW_NREST", "2"))


def cell(K, b, seed):
    tag = f"{K}_{b}_{seed}"
    fp = f"{OUT}/{tag}.json"
    if os.path.exists(fp):
        return json.load(open(fp))

    plan, inst, P0, data_by, szs, unit = BP.instance(K, seed, 100.0)
    row = BP.es_row(inst)
    byL = defaultdict(list)
    for r in inst.replicas:
        byL[row[str(r.logical)]].append(r.name)
    wl = elastic_logs_loads(len(szs), seed, n_streams=max(1, K // b))
    wl_by = {r.name: wl[row[str(r.logical)]] for r in inst.replicas}
    driver_cap(inst, P0, data_by, NODES)
    PHI = lambda pl: phi_of_placement(pl, inst, data_by, unit, wload_by=wl_by)

    inp = f"{WORK}/in_{tag}.tsv"
    with open(inp, "w") as fh:
        for L, sz in enumerate(szs):
            fh.write(f"idx-{L}\t{sz}\t{wl[L]:.6f}\n")

    outs, lines = [], []
    for i in range(NREST):
        o = f"{WORK}/o_{tag}_{i}.tsv"
        if os.path.exists(o):
            os.remove(o)
        outs.append(o)
        lines.append(f"{inp}\t{o}\t{NODES}\t{AT.COPIES - 1}\t0")
    mp = f"{WORK}/man_{tag}.tsv"
    open(mp, "w").write("\n".join(lines) + "\n")

    t0 = time.perf_counter()
    subprocess.run([ES_GRADLE, ":server:test", "--tests", "*QuboBenchDriverTests*",
                    "-Dtests.heap.size=4g", f"-Dtests.jvm.argline=-Dqubo.batch={mp}"],
                   capture_output=True, text=True, cwd=ES_REPO)
    secs = time.perf_counter() - t0

    phis, solve_ms, finish = [], [], []
    for o in outs:
        if not os.path.exists(o):
            continue
        got = defaultdict(list)
        for line in open(o):
            if line.startswith("#solve_ns"):
                solve_ms.append(int(line.split()[1]) / 1e6)
            elif line.startswith("#finish"):
                finish.append(line.split()[1])
            elif line.strip() and not line.startswith("#"):
                nm, _s, nd = line.rstrip("\n").split("\t")
                got[int(nm.split("-")[1])].append(nd.replace("node-", "node"))
        pl, ok = {}, True
        for L, nds in got.items():
            if len(byL[L]) != len(nds):
                ok = False
                break
            for nmm, nd in zip(byL[L], nds):
                pl[nmm] = nd
        if ok and len(pl) == len(inst.replicas):
            phis.append(PHI(pl))

    r = dict(k=K, backing=b, streams=max(1, K // b), seed=seed, nodes=NODES,
             phis=phis, n_distinct=len(set(round(p, 9) for p in phis)),
             phi=min(phis) if phis else None,
             solve_ms=solve_ms, finish=finish, gradle_secs=secs)
    json.dump(r, open(fp, "w"))
    print(f"  k={K:>3} b={b} seed={seed}  Phi "
          f"{r['phi'] if r['phi'] is not None else float('nan'):>10.1f}  "
          f"runs {len(phis)} distinct {r['n_distinct']}  solve "
          f"{min(solve_ms) if solve_ms else float('nan'):.0f}ms", flush=True)
    return r
