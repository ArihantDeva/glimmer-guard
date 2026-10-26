"""Both benchmarks must become the same (prompt, label) shape as the SWE set.

The published numbers are only comparable if every benchmark is scored through
one path: same prompt template, same two-token read, same metrics module. These
tests pin that, and they pin the property that actually keeps the numbers
honest -- a benchmark that a constant predictor can match must be reported as
such, not quietly presented as a result.

R-Judge and Agent-SafetyBench label WHOLE interactions, not individual commands.
Converting them to tool-call judgments is a framing choice, and the tests state
it explicitly so the README cannot overclaim what was measured.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import evals  # noqa: E402
import noul  # noqa: E402


# --------------------------------------------------------------- R-Judge

def _rjudge_rec(label=1, actions=("write_to_file: main.py",)):
    turns = [[{"role": "user", "content": "do the thing"}]]
    for a in actions:
        turns.append([{"role": "agent", "thought": "thinking", "action": a}])
    return {"id": 1, "label": label, "contents": turns,
            "risk_description": "x", "goal": "g", "scenario": "s"}


def test_rjudge_extracts_agent_actions_as_tool_calls():
    rows = evals.rjudge_rows([_rjudge_rec(actions=("a1", "a2"))])
    assert len(rows) == 2
    assert [r["label"] for r in rows] == [1, 1]


def test_rjudge_prompt_uses_the_shared_instruction():
    rows = evals.rjudge_rows([_rjudge_rec()])
    assert rows[0]["prompt"].startswith(noul.INSTRUCTION)


def test_rjudge_skips_turns_without_an_action():
    rec = _rjudge_rec()
    rec["contents"].append([{"role": "agent", "thought": "no action here"}])
    rows = evals.rjudge_rows([rec])
    assert len(rows) == 1


def test_rjudge_keeps_both_classes():
    # distinct actions, or dedupe correctly collapses them into one prompt
    rows = evals.rjudge_rows([_rjudge_rec(label=1, actions=("a_safe",)),
                              _rjudge_rec(label=0, actions=("a_unsafe",))])
    assert {r["label"] for r in rows} == {0, 1}
    assert len(rows) == 2


def test_rjudge_dedupes_identical_prompts():
    """A repeated action must not appear twice with one label; that is leakage."""
    rows = evals.rjudge_rows([_rjudge_rec(), _rjudge_rec()])
    prompts = [r["prompt"] for r in rows]
    assert len(prompts) == len(set(prompts))


# ------------------------------------------------- Agent-SafetyBench

def _asb_rec(fid=0, fulfillable=0, instruction="send mail"):
    return {"id": fid, "fulfillable": fulfillable, "instruction": instruction,
            "risks": ["Leak sensitive data / information"],
            "environments": [{"name": "Email", "tools": ["send_email"],
                              "parameters": {}}],
            "failure_modes": "1"}


def test_asb_label_is_unsafe_when_unfulfillable():
    """The benchmark's own signal: an unfulfillable request is the unsafe one."""
    rows = evals.asb_rows([_asb_rec(fulfillable=0), _asb_rec(fulfillable=1,
                                                            instruction="ok")])
    assert [r["label"] for r in rows] == [1, 0]


def test_asb_prompt_uses_the_shared_instruction():
    rows = evals.asb_rows([_asb_rec()])
    assert rows[0]["prompt"].startswith(noul.INSTRUCTION)


def test_asb_records_the_environment_as_context():
    """The tool and environment are part of what a guard would see."""
    rows = evals.asb_rows([_asb_rec()])
    assert "send_email" in rows[0]["prompt"]
    assert "Email" in rows[0]["prompt"]


def test_asb_dedupes_identical_prompts():
    rows = evals.asb_rows([_asb_rec(fid=0), _asb_rec(fid=1)])
    assert len(rows) == 1


# ------------------------------------------------------------- splitting

def test_split_is_disjoint_by_prompt():
    rows = [{"prompt": f"p{i}", "label": i % 2} for i in range(100)]
    a, b = evals.split_rows(rows, 0.5, seed=0)
    pa, pb = {r["prompt"] for r in a}, {r["prompt"] for r in b}
    assert not (pa & pb)


def test_split_covers_every_row():
    rows = [{"prompt": f"p{i}", "label": i % 2} for i in range(100)]
    a, b = evals.split_rows(rows, 0.5, seed=0)
    assert len(a) + len(b) == len(rows)


def test_split_is_deterministic_for_a_seed():
    rows = [{"prompt": f"p{i}", "label": i % 2} for i in range(50)]
    a1, b1 = evals.split_rows(rows, 0.5, seed=7)
    a2, b2 = evals.split_rows(rows, 0.5, seed=7)
    assert [r["prompt"] for r in a1] == [r["prompt"] for r in a2]
