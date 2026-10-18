#!/usr/bin/env python3
"""Index sizes following Couchbase's perfrunner test configurations (fields, key lengths, collection ratios)."""
import random

PERFRUNNER_FIELDS = {
    "category":  1,
    "year":      4,
    "city":      6,
    "realm":     6,
    "country":   6,
    "county":    6,
    "coins":     8,
    "street":    8,
    "name":      13,
    "email":     17,
    "alt_email": 17,
    "body":      None,
}

INDEX_DEF_FIELDS = ["name", "email", "body", "alt_email", "city",
                    "coins", "category", "street", "year", "realm"]

NESTED_OVERHEAD = 450
OUTLIER_A, OUTLIER_B = 2.2, 1.0
DOC_JITTER = 0.15
OUTLIER_RATE = 0.025

PLASMA_ENTRY_OVERHEAD = 114
BACK_INDEX_FACTOR = 2
DOC_KEY_DECLARED = 50
PLANNER_ENTRY_EXTRA = 6
DEFAULT_DOC_KEY_SIZE = DOC_KEY_DECLARED + PLANNER_ENTRY_EXTRA
DEFAULT_RESIDENT_RATIO = 100

PERF_ITEMS = 100_000_000
PERF_DOC_SIZE = 1024


def body_key_size(avg_doc_size=PERF_DOC_SIZE):
    if avg_doc_size <= NESTED_OVERHEAD:
        return 0.0
    base = avg_doc_size - NESTED_OVERHEAD
    e_inv_beta = (OUTLIER_A + OUTLIER_B - 1.0) / (OUTLIER_A - 1.0)
    return (1.0 - OUTLIER_RATE) * base + OUTLIER_RATE * 2048.0 * e_inv_beta


def index_data_size(sec_key_size, num_docs,
                    doc_key_size=DEFAULT_DOC_KEY_SIZE,
                    resident_ratio=DEFAULT_RESIDENT_RATIO):
    data = (PLASMA_ENTRY_OVERHEAD + sec_key_size + doc_key_size) \
        * num_docs * BACK_INDEX_FACTOR
    return data, data * resident_ratio / 100.0


RATIO_CONFIGS = {
    "even":      None,
    "uneven10":  list(range(1, 11)),
    "uneven50":  list(range(1, 51)),
    "app3":      [119, 280, 1],
}

SKEW_RAMPS = {50: 8, 100: 16, 150: 24, 200: 33, 250: 41}
for _target, _n in SKEW_RAMPS.items():
    RATIO_CONFIGS[f"ramp{_target}"] = list(range(1, _n + 1))


def perfrunner_sizes(k, seed, fields=None, items=PERF_ITEMS,
                     avg_doc_size=PERF_DOC_SIZE, ratio_config="even",
                     resident_ratio=DEFAULT_RESIDENT_RATIO):
    fields = fields or INDEX_DEF_FIELDS
    ratios = RATIO_CONFIGS[ratio_config]
    nf = len(fields)
    if ratios:
        m = max(1, min(len(ratios), k // nf or 1))
        if m == len(ratios):
            sel = list(ratios)
        else:
            step = (len(ratios) - 1) / (m - 1) if m > 1 else 0
            sel = [ratios[round(i * step)] for i in range(m)]
    else:
        sel = [1] * max(1, k // nf or 1)
    ncol = len(sel)
    tot = sum(sel) or 1
    col_docs = [items * x / tot for x in sel]

    rng = random.Random(seed)
    jit = [1.0 + rng.uniform(-DOC_JITTER, DOC_JITTER) for _ in range(ncol)]
    jsum = sum(x * j for x, j in zip(sel, jit)) or 1
    col_docs = [items * x * j / jsum for x, j in zip(sel, jit)]
    pairs = [(i % ncol, fields[(i // ncol) % len(fields)]) for i in range(k)]
    out = []
    for c, f in pairs:
        ks = body_key_size(avg_doc_size) if f == "body" else PERFRUNNER_FIELDS[f]
        _, mem = index_data_size(ks, col_docs[c], resident_ratio=resident_ratio)
        out.append((f, ks, col_docs[c], mem))
    return out
