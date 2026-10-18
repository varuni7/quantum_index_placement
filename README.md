# Leveraging Quantum(-Inspired) Accelerators for Distributed Index Placement Optimization

Code and configurations for the paper. We encode distributed index placement as a QUBO,
sample it with D-Wave's `neal`, and compare against two production planners on identical
instances, each scored by its own objective: Couchbase's index planner (`cbindexplan`)
and Elasticsearch's shard allocator. Samples that do not decode to a legal placement are
discarded; nothing is repaired.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Production code is not included; build it from the vendors' repositories:

* **Couchbase planner:** [couchbase/indexing @ f3eae1f](https://github.com/couchbase/indexing/tree/f3eae1f8165487bda9dcade741b10f00639f7d1e),
  tool in `secondary/cmd/cbindexplan` (Go 1.26). Then `export CBINDEXPLAN_BIN=/path/to/cbindexplan`.
* **Elasticsearch allocator:** [elastic/elasticsearch @ v9.5.3](https://github.com/elastic/elasticsearch/tree/v9.5.3)
  with our test harness; see [elasticsearch_harness/](elasticsearch_harness/README.md).

`--qubo-only` runs our side of any experiment without either.

Instances follow Couchbase's [perfrunner](https://github.com/couchbase/perfrunner) test
configurations, Elastic's [elastic/logs](https://github.com/elastic/rally-tracks/tree/master/elastic/logs)
Rally track, and TPC-H index sizes (`src/external/hyrise-tpch-sf10/PROVENANCE.md`).

## Experiments in the paper

| Config | Paper | Runs | Time |
|---|---|---|---|
| **`configs/full/couchbase_fresh.json`** | Sec. 7, fresh placement | 100 | ~17 h |
| **`configs/full/couchbase_rebalance_quota_down_70.json`** | Table 2, Figure 2 | 50 | ~8 h |
| **`configs/full/couchbase_rebalance_quota_up_90.json`** | Figure 3 | 50 | ~8 h |
| **`configs/full/couchbase_rebalance_node_added_90.json`** | Figure 4 | 50 | ~8 h |
| **`configs/full/couchbase_movement_limit.json`** | Figure 5 | 20 | ~2 h |
| **`configs/full/es_qubo.json`**, **`es_allocator.json`** | Table 3 | 75 each | ~8 h, ~1.5 h |
| `configs/full/couchbase_rebalance_capacity_*.json` | Sec. 7, capacity | 50 each | ~8 h each |

```bash
python run.py configs/full/couchbase_rebalance_quota_down_70.json
```

Options: `--budget S`, `--limit N`, `--qubo-only`, `--out DIR`. Results go to `results_out/`;
an interrupted config resumes where it stopped.

## Shorter verification runs

Not in the paper: a few instances from the grid above, chosen so the gap between the
methods is far larger than run-to-run variation. Each prints our value next to yours.

| Config | Time |
|---|---|
| `configs/smaller/smoke_test.json` (install check) | ~30 s |
| `configs/smaller/couchbase_fresh.json` | ~30 min |
| `configs/smaller/couchbase_rebalance.json` | ~30 min |
| `configs/smaller/es_qubo.json` (deterministic: matches exactly) | ~15 min |
| `configs/smaller/es_allocator.json` | minutes |

Couchbase runs use a 300 s wall-clock budget per method, so exact values vary with the machine.

## Where settings live

| Setting | Code |
|---|---|
| Couchbase planner objective (Eq. 19) | `src/cb_cost.py` |
| Couchbase scenarios and index sizing | `src/cb_scenarios.py`, `src/generators/cbindexplan_io.py` |
| QUBO (Section 4), weights λ, β, κ, ν, movement prices γ̂ | `src/qubo_move_variables.py`, `src/exp_cbcost.py`, `experiments/couchbase.py` |
| Elasticsearch objective Φ and its QUBO | `src/es_exact.py`, `src/es_qubo.py`, `experiments/elasticsearch.py` |
| Factored Φ checked against direct summation | `python src/phi_factored.py` |
