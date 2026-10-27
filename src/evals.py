"""Turn R-Judge and Agent-SafetyBench into the same (prompt, label) shape.

Why this module exists: the training data is (tool call -> did it fail). The two
published benchmarks are not that. R-Judge labels a whole agent interaction as
unsafe; Agent-SafetyBench labels a request as unfillable. Converting either into
a per-tool-call judgment is a FRAMING CHOICE, made explicitly here so the README
reports what was actually measured rather than implying the benchmarks are
command-level.

Every row produced here goes through noul.INSTRUCTION and is scored by the same
two-token read as training, so no benchmark gets a bespoke prompt.
"""
import json
import os
import random

import noul

D = os.environ.get("JEV_DATA",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"))


def _prompt(body):
    return f"{noul.INSTRUCTION}\n<command>\n{body}\n</command>\n\nAnswer:"


def _dedupe(rows):
    """Collapse identical prompts. A prompt that appears with both labels is
    unmatchable, so the first label wins and the count is reported."""
    seen, out, conflicts = {}, [], 0
    for r in rows:
        k = r["prompt"]
        if k in seen:
            if seen[k] != r["label"]:
                conflicts += 1
            continue
        seen[k] = r["label"]
        out.append(r)
    return out, conflicts


# --------------------------------------------------------------- R-Judge

def rjudge_rows(records):
    """One row per agent action, labelled by the interaction's safety label.

    R-Judge's label is per interaction. Reusing it for each action inside that
    interaction assumes every action in an unsafe interaction is unsafe, which
    is the benchmark's aggregate view, not ground truth per step.
    """
    rows = []
    for rec in records:
        label = int(rec["label"])
        for turn in rec.get("contents", []):
            for msg in turn:
                action = msg.get("action") if isinstance(msg, dict) else None
                if not action:
                    continue
                rows.append({
                    "prompt": _prompt(action),
                    "label": label,
                    "bench": "rjudge",
                    "src_id": rec.get("id"),
                })
    out, conflicts = _dedupe(rows)
    return out


# ------------------------------------------------- Agent-SafetyBench

def asb_rows(records):
    """One row per request; unfulfillable == the unsafe case.

    `fulfillable` is the benchmark's own field: 0 means the request cannot be
    satisfied without violating a risk category, so it is the unsafe label.
    """
    rows = []
    for rec in records:
        env = (rec.get("environments") or [{}])[0]
        tools = ", ".join(env.get("tools") or []) or "(none)"
        body = (f"environment: {env.get('name') or '(none)'}\n"
                f"tool: {tools}\n"
                f"request: {rec['instruction']}")
        rows.append({
            "prompt": _prompt(body),
            "label": 1 if int(rec["fulfillable"]) == 0 else 0,
            "bench": "asb",
            "src_id": rec.get("id"),
        })
    out, _ = _dedupe(rows)
    return out


# ------------------------------------------------------------- splitting

def split_rows(rows, frac, seed=0):
    """Split by whole prompt so no prompt appears on both sides."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = max(1, int(len(rows) * frac))
    return rows[:n], rows[n:]


def load_bench(name):
    """Training split, dev split, and full rows for a benchmark."""
    if name == "rjudge":
        recs = [json.loads(l) for l in open(f"{D}/rjudge.jsonl")]
        return rjudge_rows(recs)
    if name == "asb":
        recs = json.load(open(f"{D}/asb.json"))
        return asb_rows(recs)
    raise SystemExit(f"unknown benchmark {name!r}")
