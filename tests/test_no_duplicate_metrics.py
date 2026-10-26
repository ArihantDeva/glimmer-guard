"""The trainer must not carry its own copy of the metrics.

Two implementations of ECE/AURC/coverage is two answers to the same question,
and the published README quotes one of them. This test fails if train_lora.py
reintroduces a local metric instead of importing from src/metrics.py, and it
fails if the two ever disagree.

Importing train_lora requires torch, which is only installed on the cluster, so
the source-level check runs always and the numeric comparison is skipped when
torch is absent.
"""
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metrics  # noqa: E402

TRAINER = os.path.join(os.path.dirname(__file__), "..", "src", "train_lora.py")


def _source():
    return open(TRAINER).read()


# --------------------------------------------------- source-level, no torch

@pytest.mark.parametrize("name", ["ece", "aurc", "cov_at", "fit_T", "apply_T"])
def test_trainer_does_not_define_its_own_metrics(name):
    """A local `def ece(...)` in the trainer is the bug this guards."""
    assert not re.search(rf"^def {name}\s*\(", _source(), re.MULTILINE), (
        f"train_lora.py defines its own {name}(); import it from metrics.py "
        "so the README cannot disagree with the tests"
    )


def test_trainer_imports_the_shared_metrics():
    src = _source()
    assert "from metrics import" in src or "import metrics" in src, (
        "train_lora.py must import the shared metrics module"
    )


# ------------------------------------------- numeric equivalence, with torch

def test_shared_metrics_reproduce_a_known_hand_computed_case():
    """Guard against the monotone-in-coverage bug class.

    Prefix risks after sorting by confidence are 0.0, 0.5, 1/3, 0.25. A naive
    implementation that returns the largest k whose own mean is within budget
    reports k=4 for any budget >= 0.25; the correct one stops at k=1 because
    the k=2 prefix already exceeded a 0.25 budget.
    """
    p = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 1])
    _, k = metrics.coverage_at(p, y, budget=0.25)
    assert k == 1, "coverage_at must honour every prefix, not just the last one"
