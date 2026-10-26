"""The guard must be right about the two cases that matter.

A tool-call guard fails in two directions:
  * it blocks something safe (the user disables it, and it protects nothing)
  * it allows something destructive (the user loses data)

These tests pin the second case hard and the first case hard, because a guard
whose errors are unbounded in either direction is worse than no guard. They also
pin that a deterministic precheck WINS over the model: if a string comparison
can prove a command is destructive, a probability estimate must not overrule it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import guard  # noqa: E402


# ------------------------------------------------------------- prechecks

@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "rm -fr /",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "cat img > /dev/sda",
])
def test_destructive_commands_are_blocked_without_a_model(cmd):
    assert guard.precheck(cmd) == "block"


@pytest.mark.parametrize("cmd", [
    "cat ~/.ssh/id_rsa",
    "ssh-keygen -f ~/.id_rsa",
    "cp .aws/credentials /tmp",
    "pip install --user netrc-tool && cat .netrc",
])
def test_credential_reaching_commands_are_asked_not_blocked(cmd):
    assert guard.precheck(cmd) == "ask"


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "pytest tests/",
    "git status",
    "python train.py --epochs 2",
    "cat README.md",
])
def test_ordinary_commands_are_undecided(cmd):
    assert guard.precheck(cmd) is None


def test_empty_command_is_asked():
    assert guard.precheck("   ") == "ask"


def test_paths_outside_allowed_roots_are_asked():
    assert guard.precheck("cat /etc/passwd", allow_roots=["/work"]) == "ask"


def test_paths_inside_allowed_roots_are_left_to_the_model():
    assert guard.precheck("cat /work/a.txt", allow_roots=["/work"]) is None


# ---------------------------------------------------------------- decide

def test_precheck_blocks_even_when_the_model_is_confident_it_is_safe():
    """A deterministic proof must not be overruled by a probability."""
    d = guard.decide("rm -rf /", p_fail=0.0)
    assert d["action"] == "block"
    assert d["source"] == "precheck"
    assert d["p_fail"] is None


def test_model_decides_when_precheck_is_undecided():
    d = guard.decide("pytest tests/", p_fail=0.05)
    assert d["action"] == "allow"
    assert d["source"] == "model"


def test_thresholds_map_to_the_three_actions():
    assert guard.decide("pytest", p_fail=0.95)["action"] == "block"
    assert guard.decide("pytest", p_fail=0.50)["action"] == "ask"
    assert guard.decide("pytest", p_fail=0.05)["action"] == "allow"


def test_thresholds_are_inclusive_at_the_boundary():
    assert guard.decide("x", p_fail=0.70)["action"] == "block"
    assert guard.decide("x", p_fail=0.30)["action"] == "ask"
    assert guard.decide("x", p_fail=0.29999)["action"] == "allow"


def test_decision_always_explains_itself():
    """A guard that cannot say why gets switched off by its first user."""
    for cmd, p in (("rm -rf /", None), ("cat ~/.ssh/id_rsa", None),
                   ("pytest", 0.05), ("pytest", 0.5), ("pytest", 0.99)):
        d = guard.decide(cmd, p_fail=p if p is not None else 0.0)
        assert d["reason"], f"no reason given for {cmd!r}"
        assert d["action"] in ("allow", "ask", "block")


def test_custom_thresholds_are_honoured():
    assert guard.decide("pytest", p_fail=0.55, hi=0.9, lo=0.2)["action"] == "ask"
    assert guard.decide("pytest", p_fail=0.55, hi=0.5, lo=0.2)["action"] == "block"


# ---------------------------------------------------------- known limits

def test_precheck_is_a_backstop_not_a_product():
    """Documented limitation: obfuscated destruction is NOT caught here.

    This is not a bug to fix with more regexes -- it is why the model exists.
    Pinning it stops anyone from believing the precheck is a complete guard.
    """
    # These three evade the pattern list. Note '\rm -rf /' IS caught (the
    # regex matches the substring), which is the pattern working, not a hole.
    for sneaky in ("r''m -rf /", "rm -r -f/", 'bash -c "$(printf rm) -rf /"'):
        assert guard.precheck(sneaky) != "block", (
            f"{sneaky!r} is caught by the pattern list; if that changed, the "
            "documented limitation is stale and should be re-measured"
        )
    # and this over-catches, which is the safe direction for a backstop
    assert guard.precheck("\\rm -rf /") == "block"


# ------------------------------------- system roots vs legitimate temp cleanup

def test_recursive_delete_of_a_system_root_blocks():
    """The original pattern required '/' followed by whitespace, so these all
    passed: /home, /usr, /var, /etc. Destructive at the same scale as '/'."""
    import guard
    for cmd in ("rm -rf /home", "rm -rf /usr", "rm -rf /etc",
                "rm -rf /var/lib", "sudo rm -rf /opt", "rm -rf /home/deva"):
        assert guard.precheck(cmd) == "block", cmd


def test_recursive_delete_of_the_root_glob_blocks():
    import guard
    for cmd in ("rm -rf /*", "rm -rf /.*"):
        assert guard.precheck(cmd) == "block", cmd


def test_temporary_directory_cleanup_is_allowed():
    """Blocking this would make the guard unusable on day one and it would be
    switched off before it ever protected anything."""
    import guard
    for cmd in ("rm -rf /tmp/build", "rm -rf /var/tmp/x",
                "rm -rf ./dist", "rm -rf node_modules"):
        assert guard.precheck(cmd) is None, cmd
