#!/usr/bin/env bash
# Full local gate. Fails loudly on the first broken invariant.
# This is the command the README tells a reviewer to run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "== unit tests =="
"$PY" -m pytest tests/ -q

echo
echo "== dataset integrity =="
"$PY" - <<'PYEOF'
import json, sys
from collections import Counter

required = {"train", "dev", "test"}
seen = {}
for split in required:
    path = f"data/swe_{split}.jsonl"
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        sys.exit(f"MISSING {path} - run: python3 src/fetch_swe_pairs.py")
    if not rows:
        sys.exit(f"EMPTY {path}")
    for r in rows:
        for k in ("prompt", "label"):
            if k not in r:
                sys.exit(f"{path} row missing '{k}'")
        if r["label"] not in (0, 1):
            sys.exit(f"{path} bad label {r['label']}")
    seen[split] = rows
    fail = sum(r["label"] for r in rows)
    print(f"  {split:5s} n={len(rows):6d}  fail={fail:5d} ({fail/len(rows)*100:5.1f}%)")

# no split leakage: prompts must be disjoint
tr = {r["prompt"] for r in seen["train"]}
te = {r["prompt"] for r in seen["test"]}
dv = {r["prompt"] for r in seen["dev"]}
assert not (tr & te), f"{len(tr & te)} prompts leaked train->test"
assert not (tr & dv), f"{len(tr & dv)} prompts leaked train->dev"
assert not (dv & te), f"{len(dv & te)} prompts leaked dev->test"
print("  splits disjoint: OK")

# label classes present
cls = Counter(r.get("cls") for r in seen["train"])
assert len([c for c in cls if c != "ok"]) >= 3, f"too few failure classes: {cls}"
print(f"  failure classes: {len(cls)-1} (+ok)")
PYEOF

echo
echo "== floor test (must not be separable by surface form) =="
"$PY" src/floor.py data/swe_train.jsonl data/swe_test.jsonl | tail -3

echo
echo "== benchmark floor tests (R-Judge, Agent-SafetyBench) =="
"$PY" src/bench_floor.py

echo
echo "ALL CHECKS PASSED"
