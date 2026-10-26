"""A shell wrapper acts on the exit code, so the exit code is the contract.

Testing the JSON alone would leave the actually-used channel unverified: a
wrapper sees `$?`, not stdout. Each test drives main() and asserts the number a
shell would branch on.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import muse_guard_cli  # noqa: E402


def test_destructive_command_exits_3(tmp_path, capsys):
    caplog = None
    rc = muse_guard_cli.main(["--no-model", "rm -rf /etc"])
    assert rc == 3
    assert '"block"' in capsys.readouterr().out


def test_key_access_exits_2(capsys):
    rc = muse_guard_cli.main(["--no-model", "cat ~/.ssh/id_rsa"])
    assert rc == 2
    assert '"ask"' in capsys.readouterr().out


def test_ordinary_command_without_a_model_exits_0(capsys):
    rc = muse_guard_cli.main(["--no-model", "uv run pytest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"precheck_only"' in out and '"allow"' in out


def test_exit_codes_are_distinct_and_nonzero_for_escalation():
    codes = muse_guard_cli.EXIT
    assert codes["allow"] == 0
    assert codes["ask"] != 0 and codes["block"] != 0
    assert codes["ask"] != codes["block"], (
        "a wrapper must be able to tell ask from block")


def test_unreachable_server_exits_4_and_says_why(capsys):
    """Fail closed. Exit 0 here would let a dead guard look like a clear one."""
    rc = muse_guard_cli.main(
        ["--base-url", "http://127.0.0.1:1", "ls -la"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "cannot decide" in err


def test_unknown_server_error_is_not_reported_as_allow(capsys):
    rc = muse_guard_cli.main(["--base-url", "http://127.0.0.1:1", "echo hi"])
    assert rc != 0


def test_json_is_one_line_for_greppable_logs(capsys):
    muse_guard_cli.main(["--no-model", "ls"])
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0 and out.startswith("{")
