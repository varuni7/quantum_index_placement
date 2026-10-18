#!/usr/bin/env python3
"""Write loads from Elastic's elastic/logs Rally track (integration_ratios)."""
import random

DEFAULT_MEAN = 0.25


ELASTIC_LOGS_RATIOS = [
    0.500,
    0.250,
    0.050,
    0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
    0.010, 0.010, 0.010, 0.010, 0.010,
]


def elastic_logs_loads(k, seed, n_streams=16, mean=DEFAULT_MEAN):
    rng = random.Random(seed)
    n = min(n_streams, k)
    reps_needed = -(-n // len(ELASTIC_LOGS_RATIOS))
    ratios = (ELASTIC_LOGS_RATIOS * reps_needed)[:n]
    picks = rng.sample(range(k), n)
    scale = (k * mean) / sum(ratios)
    out = [0.0] * k
    for i, r in zip(picks, ratios):
        out[i] = r * scale
    return out
