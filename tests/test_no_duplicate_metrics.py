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


def test_summarize_returns_only_scalars():
    """summarize() is formatted directly into reports; a tuple would crash."""
    import numpy as np
    p = np.array([0.9, 0.4, 0.6, 0.1])
    y = np.array([1, 0, 1, 0])
    out = metrics.summarize(p, y)
    for k, v in out.items():
        assert isinstance(v, (int, float)), (
            f"summarize()[{k!r}] is {type(v).__name__}, not a scalar; the "
            "report formatter will raise TypeError on the cluster"
        )


def test_coverage_helper_returns_a_pair_but_summarize_unwraps_it():
    import numpy as np
    p = np.array([0.9, 0.4])
    y = np.array([1, 0])
    cov, n = metrics.coverage_at(p, y, 0.05)
    assert isinstance(cov, float) and isinstance(n, int)
    assert metrics.summarize(p, y)["cov5"] == cov


def test_no_variable_is_read_outside_its_scope():
    """Catch 'NameError: name X is not defined' before it costs a GPU run.

    train_glimmer.py read `loss` after the accumulation loop that defined it had
    closed. Nothing static caught it; it surfaced 12 minutes into a cluster job.
    This walks every function in the training scripts and flags a load of a name
    that is assigned nowhere in that function, in the module, or in builtins.
    """
    import ast
    import builtins

    here = os.path.dirname(__file__)
    targets = [os.path.join(here, "..", "src", f)
               for f in ("train_glimmer.py", "train_lora.py", "rescore.py")]
    problems = []
    for path in targets:
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path).read())
        module_names = set(dir(builtins))
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                module_names.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    module_names.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                module_names.add(n.name)
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            local = set(module_names) | {a.arg for a in fn.args.args}
            local |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    local.add(n.id)
                elif isinstance(n, (ast.comprehension,)):
                    for t in ast.walk(n.target):
                        if isinstance(t, ast.Name):
                            local.add(t.id)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    if n.id not in local:
                        problems.append(f"{os.path.basename(path)}:{fn.name}"
                                        f" reads unbound '{n.id}'")
    assert not problems, "\n".join(sorted(set(problems)))
