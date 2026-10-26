"""Calibration and selective-prediction metrics.

Every published number in the README comes from these functions, so each one
is small, pure, and unit-tested against a hand-computed value (see
tests/test_metrics.py). Nothing here imports torch: the metrics operate on
numpy arrays so they can be checked without a GPU.
"""
import numpy as np


def brier(p, y):
    """Mean squared error of the predicted probability."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _confidence(p):
    return np.maximum(p, 1 - p)


def _correct(p, y):
    return ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).astype(float)


def ece(p, y, bins=15):
    """Expected calibration error over `bins` equal-width confidence bins."""
    p, y = np.asarray(p, float), np.asarray(y)
    conf, correct = _confidence(p), _correct(p, y)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else edges[i + 1] + 1e-12
        m = (conf >= edges[i]) & (conf < hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def aurc(p, y):
    """Area under the risk-coverage curve. Lower is better; 0 is perfect."""
    p, y = np.asarray(p, float), np.asarray(y)
    wrong = 1.0 - _correct(p, y)
    order = np.argsort(-_confidence(p), kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(wrong) + 1)
    return float(np.mean(risk))


def coverage_at(p, y, budget=0.05):
    """Largest prefix (by descending confidence) whose error rate stays within
    `budget`. Returns (coverage_fraction, n_items).

    This is the operational number for a guard: it answers "how often may I
    trust the model at a k% error budget?" rather than "how often is it right?"
    """
    p, y = np.asarray(p, float), np.asarray(y)
    wrong = (1.0 - _correct(p, y)).astype(int)
    order = np.argsort(-_confidence(p), kind="stable")
    w = wrong[order]
    # The risk-coverage curve is NOT monotone: adding items can lower the
    # running mean. A budget is only honoured if EVERY prefix up to k is
    # within budget, so track the running maximum and stop at the first
    # violation. Returning the largest k whose own mean is within budget
    # would silently overstate coverage.
    running_max = 0.0
    best = 0
    for k in range(1, len(w) + 1):
        running_max = max(running_max, w[:k].mean())
        if running_max <= budget:
            best = k
        else:
            break
    return best / len(w), best


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def apply_temperature(p, t):
    """Rescale probabilities by a single temperature."""
    return 1.0 / (1.0 + np.exp(-_logit(p) / float(t)))


def fit_temperature(p, y, lo=0.25, hi=20.0, step=0.05):
    """Fit one scalar temperature on held-out data by minimising log-loss."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    lg = _logit(p)
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(lo, hi + 1e-9, step):
        q = np.clip(1.0 / (1.0 + np.exp(-lg / t)), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def gate(p, y, hi=0.7, lo=0.3):
    """Turn p(fail) into a three-way decision and score it.

    block  : p >= hi        refuse to run
    ask    : lo <= p < hi   surface to the human
    allow  : p <  lo        run unattended

    The thresholds live in code, not in the model: the model supplies a
    probability, the caller owns the policy.
    """
    p, y = np.asarray(p, float), np.asarray(y)
    block = p >= hi
    ask = (p >= lo) & (p < hi)
    allow = p < lo
    fails = int((y == 1).sum())
    caught = int((block & (y == 1)).sum())
    n_safe = int((y == 0).sum())
    return {
        "block_rate": float(block.mean()),
        "ask_rate": float(ask.mean()),
        "allow_rate": float(allow.mean()),
        "fail_recall": caught / fails if fails else 0.0,
        "block_precision": caught / int(block.sum()) if block.sum() else 0.0,
        "false_block_rate": int((block & (y == 0)).sum()) / n_safe if n_safe else 0.0,
    }


def summarize(p, y):
    """Every metric reported in the README, as one dict."""
    p, y = np.asarray(p, float), np.asarray(y)
    return {
        "n": int(len(y)),
        "acc": float((_correct(p, y)).mean()),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "aurc": aurc(p, y),
        "cov5": coverage_at(p, y, 0.05)[0],
        "cov10": coverage_at(p, y, 0.10)[0],
        "cov20": coverage_at(p, y, 0.20)[0],
    }
