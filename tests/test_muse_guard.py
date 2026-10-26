"""The integration is only real if a crash is visible.

Most of these tests are about ordering and failure, not accuracy: the model is a
stub. What is being verified is that Muse Code gets a decision it can act on and
that the guard cannot fail open.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import muse_guard  # noqa: E402


class StubScorer:
    def __init__(self, p):
        self.p = p
        self.seen = []

    def score(self, command):
        self.seen.append(command)
        return self.p


# --------------------------------------------------------- ordering

def test_destructive_command_is_blocked_without_calling_the_model():
    """A string comparison must not have to win an argument with a probability."""
    s = StubScorer(0.0)
    g = muse_guard.MuseGuard(s)
    d = g.inspect("rm -rf /")
    assert d["action"] == "block"
    assert d["source"] == "precheck"
    assert s.seen == [], "the model was consulted for a command a regex proves"


def test_model_cannot_override_a_precheck_block():
    s = StubScorer(0.001)          # maximally confident 'safe'
    d = muse_guard.MuseGuard(s).inspect("mkfs.ext4 /dev/sda1")
    assert d["action"] == "block"
    assert s.seen == []


def test_ssh_key_access_asks_without_calling_the_model():
    s = StubScorer(0.0)
    d = muse_guard.MuseGuard(s).inspect("cat ~/.ssh/id_rsa")
    assert d["action"] == "ask"
    assert d["source"] == "precheck"
    assert s.seen == []


def test_ordinary_command_reaches_the_model():
    s = StubScorer(0.9)
    d = muse_guard.MuseGuard(s).inspect("uv run pytest tests/")
    assert s.seen == ["uv run pytest tests/"]
    assert d["source"] == "model"
    assert d["action"] == "block"
    assert d["p_fail"] == pytest.approx(0.9)


# ------------------------------------------------------- decisions

def test_low_probability_allows():
    d = muse_guard.MuseGuard(StubScorer(0.05)).inspect("ls -la")
    assert d["action"] == "allow"


def test_mid_probability_asks():
    d = muse_guard.MuseGuard(StubScorer(0.5)).inspect("npm install")
    assert d["action"] == "ask"


def test_decision_carries_a_reason():
    d = muse_guard.MuseGuard(StubScorer(0.5)).inspect("npm install")
    assert "0.5" in d["reason"] and "lo=" in d["reason"]


def test_decision_is_serialisable_for_a_run_log():
    d = muse_guard.MuseGuard(StubScorer(0.5)).inspect("npm install")
    line = muse_guard.MuseGuard().to_json(d)
    assert line.startswith("{") and "\n" not in line


# ------------------------------------------------- failure behaviour

class BrokenScorer:
    def score(self, command):
        raise RuntimeError("llama-server is down")


def test_a_broken_scorer_raises_rather_than_allowing():
    """Fail-open is the one unacceptable behaviour: it reports safety it never
    established."""
    with pytest.raises(RuntimeError, match="llama-server is down"):
        muse_guard.MuseGuard(BrokenScorer()).inspect("ls -la")


def test_without_a_scorer_undecided_commands_are_labelled_not_guessed():
    d = muse_guard.MuseGuard().inspect("ls -la")
    assert d["p_fail"] is None
    assert d["source"] == "precheck_only"


def test_without_a_scorer_a_block_still_blocks():
    d = muse_guard.MuseGuard().inspect("dd of=/dev/sda")
    assert d["action"] == "block"


def test_empty_command_asks():
    d = muse_guard.MuseGuard(StubScorer(0.0)).inspect("   ")
    assert d["action"] == "ask"


# ------------------------------------------------------------ latency

def test_inspect_does_no_extra_scoring_work():
    """One tool call is one forward pass. Two would double the latency budget."""
    s = StubScorer(0.4)
    muse_guard.MuseGuard(s).inspect("echo hi")
    assert len(s.seen) == 1


def test_base_url_is_overridable_for_a_remote_server():
    os.environ["MUSE_GUARD_BASE_URL"] = "http://10.0.0.5:9999"
    try:
        assert muse_guard.base_url_from_env() == "http://10.0.0.5:9999"
    finally:
        del os.environ["MUSE_GUARD_BASE_URL"]


def test_base_url_default_is_local():
    os.environ.pop("MUSE_GUARD_BASE_URL", None)
    assert muse_guard.base_url_from_env().startswith("http://127.0.0.1")
