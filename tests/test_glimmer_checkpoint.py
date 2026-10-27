"""The Glimmer run must survive being killed.

The first QLoRA run projected 3.82 hours inside a 4-hour wall clock and saved
its adapter ONLY on the final step. A timeout would therefore have destroyed
several hours of GPU with nothing to show. These tests pin the fix.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_util import checkpoint_due, steps_for  # noqa: E402


def test_saves_on_the_interval():
    assert checkpoint_due(100, 677, 100) is True


def test_does_not_save_off_the_interval():
    assert checkpoint_due(101, 677, 100) is False


def test_always_saves_on_the_final_step_even_if_the_interval_misses_it():
    """677 % 100 != 0, so without this the run ends having saved nothing since
    step 600."""
    assert checkpoint_due(677, 677, 100) is True


def test_saves_on_the_first_step():
    assert checkpoint_due(1, 677, 1) is True


def test_zero_interval_is_rejected_rather_than_silently_never_saving():
    with pytest.raises(ValueError, match="positive"):
        checkpoint_due(1, 10, 0)


def test_negative_interval_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        checkpoint_due(1, 10, -5)


def test_a_one_step_run_saves():
    assert checkpoint_due(1, 1, 100) is True


# ------------------------------------------------------- step arithmetic

def test_steps_cover_the_data():
    """5,412 train rows at bs=2 x accum=4 = 8 per step -> 677 steps."""
    assert steps_for(5412, 2, 4, 1.0) == 677


def test_steps_round_up_so_the_tail_is_not_dropped():
    assert steps_for(9, 2, 4, 1.0) == 2


def test_partial_epochs_shrink_the_run():
    assert steps_for(1000, 2, 4, 0.5) == 63


def test_zero_batch_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        steps_for(100, 0, 4, 1.0)


# ------------------------------------------- batch-script exit propagation

def test_shell_scripts_do_not_swallow_child_failures():
    """A loop that ignores its children's exit codes reports success however
    badly they failed. The first bench job reported COMPLETED exit 0 while all
    three seeds had crashed on a NameError; only an implausible 49-second
    runtime gave it away. The exit code should have been the alarm.

    Static check: every multi-step run script must propagate or fail fast.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    offenders = []
    for sh in sorted(root.glob("run_*.sh")):
        text = sh.read_text()
        if "for " not in text:
            continue
        guards = ("set -e", "|| exit", "FAILED", "exit \"$rc\"", "exit $rc")
        if not any(g in text for g in guards):
            offenders.append(sh.name)
    assert not offenders, (
        "these scripts loop without propagating failure: " + ", ".join(offenders))
