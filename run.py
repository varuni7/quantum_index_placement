#!/usr/bin/env python3
"""Run one experiment config.

    python run.py configs/smaller/smoke_test.json               # install check, ~30 s
    python run.py configs/smaller/couchbase_rebalance.json      # ~30 min
    python run.py configs/full/couchbase_rebalance_quota_down_70.json
    python run.py CONFIG --budget 30 --limit 2 --qubo-only

One JSON result per run is written to results_out/<folder>_<config>/; runs already there
are skipped, so a config can resume. Configs in configs/smaller/ carry our values and
print them next to yours.
"""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--budget", type=float, help="override the per-method time budget (s)")
ap.add_argument("--limit", type=int, help="run only the first N runs")
ap.add_argument("--qubo-only", action="store_true", help="skip the production planner")
ap.add_argument("--out", default=os.path.join(HERE, "results_out"))
a = ap.parse_args()

cfg = json.load(open(a.config))
name = os.path.basename(os.path.dirname(os.path.abspath(a.config))) + "_" + os.path.splitext(os.path.basename(a.config))[0]
outdir = os.path.join(a.out, name); os.makedirs(outdir, exist_ok=True)
exp, params, cells = cfg["experiment"], dict(cfg.get("params", {})), cfg["cells"]
if a.budget is not None and "budget" in params:
    params["budget"] = a.budget
if a.limit:
    cells = cells[: a.limit]
if cfg.get("about"):
    print(cfg["about"])

import couchbase as CB, elasticsearch as ES
FN = {"couchbase_fresh": CB.fresh_cell, "couchbase_rebalance": CB.rebalance_cell,
      "couchbase_budget": CB.budget_cell, "es_qubo": ES.qubo_cell,
      "es_allocator": ES.allocator_cell}[exp]


def summary(r):
    if exp == "couchbase_fresh":
        return {"planner": r.get("planner", {}).get("final"), "qubo": r["qubo"]["final"]}
    if exp == "couchbase_rebalance":
        p = r.get("planner", {}).get("pts") or [[None] * 4]
        q = r["qubo"]["pts"] or [[None] * 4]
        return {"planner": p[-1][1], "qubo": q[-1][1],
                "planner_moved_pct": p[-1][3], "qubo_moved_pct": q[-1][3]}
    if exp == "es_qubo":
        return {"qubo_sqrt_phi": math.sqrt(r["phi"]) if r.get("phi") else None}
    if exp == "es_allocator":
        return {"allocator_sqrt_phi": math.sqrt(r["phi"]) if r.get("phi") else None}
    return {}


def fmt(v):
    return "  --  " if v is None else (f"{v:.2f}" if abs(v) >= 1 else f"{v:.5f}")


rows = []
for i, c in enumerate(cells):
    ref = c.get("reference")
    c = {k: v for k, v in c.items() if k != "reference"}
    kw = {**params, **c}
    if a.qubo_only and exp in ("couchbase_fresh", "couchbase_rebalance"):
        kw["qubo_only"] = True
    fp = os.path.join(outdir, "_".join(f"{v}" for v in c.values()) + ".json")
    if os.path.exists(fp):
        r = json.load(open(fp))
    else:
        print(f"[{i + 1}/{len(cells)}] {exp} {c}", flush=True)
        r = FN(**kw)
        json.dump(r, open(fp, "w"), indent=1)
    if ref:
        rows.append((c, ref, summary(r)))

if rows:
    print("\n" + "metric".ljust(22) + "our run".rjust(12) + "your run".rjust(12) + "   (lower is better)")
    for c, ref, got in rows:
        print(" ".join(f"{k}={v}" for k, v in c.items()))
        for key, v in ref.items():
            if key in got:
                print("  " + key.ljust(20) + fmt(v).rjust(12) + fmt(got[key]).rjust(12))
print(f"\ndone: {outdir}")
