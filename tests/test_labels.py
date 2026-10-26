"""Unit tests for the deterministic label rules.

Every rule here is a pure function over text. No model, no LLM, no
randomness. If a rule cannot be expressed deterministically it does not
belong in this module.

Run: python3 -m pytest tests/test_labels.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from labels import (  # noqa: E402
    ERROR_CLASSES,
    classify_failure,
    is_error,
    split_command,
)


# ---------------------------------------------------------------- is_error

def test_traceback_is_error():
    assert is_error('Traceback (most recent call last):\n  File "x.py", line 1') is True


def test_explicit_error_prefix_is_error():
    assert is_error("ERROR: could not resolve host: example.com") is True


def test_permission_denied_is_error():
    assert is_error("bash: /root/x: Permission denied") is True


def test_plain_success_is_not_error():
    assert is_error("total 8\ndrwxr-xr-x 4 501 staff 128 .\n-rw-r--r-- 1 calc.py") is False


def test_command_not_found_is_error():
    assert is_error("bash: line 1: foo: command not found") is True


def test_empty_observation_is_not_error():
    """An empty observation means the tool printed nothing - not a failure."""
    assert is_error("") is False
    assert is_error("   \n  ") is False


def test_word_error_inside_prose_is_not_an_error_marker():
    """Guard against substring matches: 'error' appearing mid-sentence in
    normal output must not flip the label."""
    assert is_error("The function returns the error message string to the caller.") is False


# --------------------------------------------------------- classify_failure

def test_import_error_class():
    obs = 'Traceback (most recent call last):\nImportError: No module named numpy'
    assert classify_failure(obs) == "import_error"


def test_module_not_found_class():
    assert classify_failure("ModuleNotFoundError: No module named 'pyyaml'") == "import_error"


def test_permission_class():
    assert classify_failure("bash: /etc/shadow: Permission denied") == "permission_denied"


def test_not_found_class():
    assert classify_failure("bash: line 1: frobnicate: command not found") == "not_found"


def test_syntax_class():
    obs = 'File "a.py", line 2\n    return a-b)\n           ^\nSyntaxError: unmatched'
    assert classify_failure(obs) == "syntax_error"


def test_timeout_class():
    assert classify_failure("Terminated\nslurmstepd: error: *** JOB CANCELLED ***") == "timeout"
    assert classify_failure("Operation timed out after 30 seconds") == "timeout"


def test_runtime_error_class():
    obs = 'Traceback (most recent call last):\nValueError: bad value'
    assert classify_failure(obs) == "runtime_error"


def test_success_classifies_as_ok():
    assert classify_failure("all 12 tests passed") == "ok"


def test_every_return_is_a_known_class():
    for obs in ["", "ok", "Traceback (most recent call last):\nKeyError: 'x'",
                "Permission denied", "command not found", "SyntaxError: bad"]:
        assert classify_failure(obs) in ERROR_CLASSES


def test_classes_are_stable_and_unique():
    assert len(ERROR_CLASSES) == len(set(ERROR_CLASSES))
    assert "ok" in ERROR_CLASSES


# ------------------------------------------------------------ split_command

def test_extracts_fenced_command():
    text = "Let me look.\n\n```\nls -la\npwd\n```"
    assert split_command(text) == "ls -la\npwd"


def test_no_fence_returns_none():
    assert split_command("I will examine the file now.") is None


def test_picks_the_first_fence():
    text = "```\nfirst\n```\nthen\n```\nsecond\n```"
    assert split_command(text) == "first"


def test_multiline_and_whitespace_preserved():
    text = "```\nmake test \\\n  && echo done\n```"
    assert "&& echo done" in split_command(text)


# ---------------------------------------------------------------- pairing

def test_pairing_shape_is_enforced():
    """The dataset builder must pair ai->user only, never user->user."""
    from labels import pair_steps

    traj = [
        {"role": "user", "text": "the issue"},
        {"role": "ai", "text": "```\nls\n```"},
        {"role": "user", "text": "file1 file2"},
        {"role": "ai", "text": "Found it, no command needed."},
        {"role": "user", "text": "ok"},
        {"role": "ai", "text": "```\ncat a.py\n```"},
        {"role": "user", "text": "print(1)"},
    ]
    pairs = pair_steps(traj)
    assert len(pairs) == 2, "only ai messages carrying a fenced command produce pairs"
    assert pairs[0]["command"] == "ls"
    assert pairs[0]["observation"] == "file1 file2"
    assert pairs[1]["command"] == "cat a.py"
    assert pairs[1]["observation"] == "print(1)"


def test_pairing_drops_trailing_command_without_observation():
    from labels import pair_steps

    traj = [
        {"role": "ai", "text": "```\nls\n```"},
    ]
    assert pair_steps(traj) == []


def test_pairing_labels_are_deterministic():
    from labels import pair_steps

    traj = [
        {"role": "ai", "text": "```\nls\n```"},
        {"role": "user", "text": "Traceback (most recent call last):\nImportError: x"},
    ]
    a = pair_steps(traj)[0]
    assert a["is_error"] is True
    assert a["failure_class"] == "import_error"
