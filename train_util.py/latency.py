"""The latency number has to be measured, and it has to be the right number.

The goal requires "measured added latency p50/p95" for the guard. Three things
make that easy to get wrong:

  1. timing the model but not the HTTP round trip, or vice versa, and calling
     the result "added latency" for a guard that needs both
  2. reporting the mean. Latency is skewed; a mean hides the tail that actually
     annoys a user, so p50 and p95 are reported and the mean is not.
  3. including warmup. The first call loads weights and allocates; folding that
     into the percentile makes a fast guard look slow. Warmup is dropped and
     the drop is stated.

This module is pure timing maths plus a thin harness, so the statistics are
tested without a server.
"""
import math
import statistics
import time


def percentiles(samples, ps=(50, 95, 99)):
    """Return {p: value_ms} for each percentile. Nearest-rank, no interpolation.

    Nearest-rank is used deliberately: with a few hundred samples an
    interpolated percentile invents a value between two real measurements, and
    for a latency claim the real measurement is the one to quote.
    """
    if not samples:
        raise ValueError("no samples")
    xs = sorted(samples)
    out = {}
    for p in ps:
        if not 0 <= p <= 100:
            raise ValueError(f"percentile out of range: {p}")
        # nearest-rank: rank = ceil(p/100 * n), then index rank-1
        rank = math.ceil(p / 100.0 * len(xs))
        idx = max(0, min(len(xs) - 1, rank - 1))
        out[p] = xs[idx]
    return out


def summarize_ms(samples, warmup=0):
    """Percentile summary in milliseconds, dropping `warmup` leading samples."""
    xs = list(samples)[warmup:]
    if not xs:
        raise ValueError(f"all {len(samples)} samples were consumed by warmup")
    pct = percentiles(xs)
    return {
        "n": len(xs),
        "dropped_warmup": warmup,
        "p50_ms": pct[50],
        "p95_ms": pct[95],
        "p99_ms": pct[99],
        "min_ms": min(xs),
        "max_ms": max(xs),
        "stdev_ms": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def time_calls(fn, inputs, warmup=3, clock=time.perf_counter):
    """Call `fn` once per input and return per-call milliseconds.

    The clock is injected so the harness itself is testable.
    """
    samples = []
    for i, x in enumerate(inputs):
        t0 = clock()
        fn(x)
        dt = (clock() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)
    return samples


def added_latency_ms(baseline_fn, guarded_fn, inputs, warmup=3,
                     clock=time.perf_counter):
    """The number the goal asks for: how much the guard ADDS.

    Both paths are measured on the same inputs so the difference is not
    contaminated by different work. Returns (guarded_summary, base_summary,
    delta_p50, delta_p95).
    """
    g = summarize_ms(time_calls(guarded_fn, inputs, warmup, clock))
    b = summarize_ms(time_calls(baseline_fn, inputs, warmup, clock))
    return g, b, g["p50_ms"] - b["p50_ms"], g["p95_ms"] - b["p95_ms"]
