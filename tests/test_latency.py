"""Latency claims are easy to get wrong in specific, predictable ways.

Each test here pins one of them: the percentile must be a real measurement, the
mean must not be what gets quoted, warmup must be excluded, and "added" latency
must actually be a difference rather than a single number wearing that label.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import latency  # noqa: E402


# ------------------------------------------------------- percentiles

def test_p50_of_an_odd_sample_is_the_middle_value():
    assert latency.percentiles([1, 2, 3, 4, 5])[50] == 3


def test_percentiles_are_always_real_measurements():
    """No interpolation: every reported value must be an observed sample."""
    xs = [10, 20, 30, 40, 50, 60, 70]
    got = latency.percentiles(xs, (0, 25, 50, 95, 100))
    assert set(got.values()) <= set(xs), got


def test_p100_is_the_maximum():
    assert latency.percentiles([5, 1, 9, 3], (100,))[100] == 9


def test_p0_is_the_minimum():
    assert latency.percentiles([5, 1, 9, 3], (0,))[0] == 1


def test_percentiles_are_monotone():
    xs = list(range(1, 101))
    p = latency.percentiles(xs, (10, 50, 90, 99))
    assert p[10] <= p[50] <= p[90] <= p[99]


def test_single_sample_works():
    p = latency.percentiles([7])
    assert p[50] == 7 and p[95] == 7


def test_empty_samples_raise():
    with pytest.raises(ValueError, match="no samples"):
        latency.percentiles([])


def test_out_of_range_percentile_raises():
    with pytest.raises(ValueError, match="out of range"):
        latency.percentiles([1, 2], (150,))


def test_a_slow_tail_moves_p95_but_not_p50():
    """This is why p50 and p95 are both reported.

    Note the tail must exceed 5% of the sample for nearest-rank p95 to move:
    with exactly 5 slow values out of 100 the 95th smallest is still the last
    fast one, so p95 stays at the fast value. That is correct behaviour, not a
    bug -- use 10% here.
    """
    fast = [10] * 100
    slow = [10] * 90 + [500] * 10
    assert latency.percentiles(fast)[50] == latency.percentiles(slow)[50]
    assert latency.percentiles(slow)[95] > latency.percentiles(fast)[95]


# --------------------------------------------------------- summarize

def test_warmup_is_dropped_from_the_summary():
    """The first call pays weight loading; folding it in misreports the guard."""
    samples = [900.0] + [10.0] * 20
    s = latency.summarize_ms(samples, warmup=1)
    assert s["p50_ms"] == 10.0
    assert s["max_ms"] == 10.0
    assert s["dropped_warmup"] == 1
    assert s["n"] == 20


def test_warmup_consuming_everything_raises():
    with pytest.raises(ValueError, match="warmup"):
        latency.summarize_ms([1.0, 2.0], warmup=2)


def test_summary_reports_percentiles_not_a_mean():
    s = latency.summarize_ms([1.0] * 90 + [1000.0] * 10)
    assert "p50_ms" in s and "p95_ms" in s
    assert "mean_ms" not in s, (
        "latency is skewed; quoting a mean hides the tail that users feel")
    assert s["p50_ms"] == 1.0
    assert s["p95_ms"] == 1000.0


# ------------------------------------------------------------ timing

def _fake_clock(values):
    it = iter(values)
    return lambda: next(it)


def test_time_calls_returns_one_sample_per_input_after_warmup():
    fn = lambda x: None                                    # noqa: E731
    clock = _fake_clock([0.0, 0.010, 0.0, 0.020, 0.0, 0.030])
    ms = latency.time_calls(fn, ["a", "b", "c"], warmup=0, clock=clock)
    assert ms == pytest.approx([10.0, 20.0, 30.0])


def test_time_calls_skips_the_warmup_calls():
    clock = _fake_clock([0.0, 9.0, 0.0, 0.010, 0.0, 0.020])
    ms = latency.time_calls(lambda x: None, ["a", "b", "c"], warmup=1,
                            clock=clock)
    assert ms == pytest.approx([10.0, 20.0])


def test_added_latency_is_a_difference_not_a_single_measurement():
    """'added latency' must subtract the unguarded path."""
    calls = {"n": 0}

    def clock():
        calls["n"] += 1
        return 0.0 if calls["n"] % 2 else 0.050

    g, b, d50, d95 = latency.added_latency_ms(
        lambda x: None, lambda x: None, ["a", "b"], warmup=0, clock=clock)
    assert g["p50_ms"] == pytest.approx(50.0)
    assert b["p50_ms"] == pytest.approx(50.0)
    assert d50 == pytest.approx(0.0)
    assert d95 == pytest.approx(0.0)
