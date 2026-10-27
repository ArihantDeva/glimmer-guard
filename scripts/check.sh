#!/usr/bin/env bash
# Full local gate. Fails loudly on the first broken invariant.
# This is the command the README tells a reviewer to run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "== dataset (built from source if absent) =="
# The gate used to assert the dataset existed and tell the reader to go fetch
# it, so the one command the README names as proof failed on the first run in a
# fresh clone. Building it here is what makes "reproduces every number from a
# clean checkout" true rather than aspirational. Network is needed only when the
# files are missing; the build is deterministic given the upstream data.
if [ ! -f data/swe_pairs.jsonl ] || [ ! -f data/rjudge.jsonl ]; then
  echo "  data missing; fetching upstream (needs network)"
  "$PY" src/fetch_swe_pairs.py
  "$PY" src/fetch_rjudge.py
fi
if [ ! -f data/asb.json ]; then
  "$PY" src/fetch_asb.py
fi
if [ ! -f data/swe_train.jsonl ]; then
  echo "  building leak-free splits"
  "$PY" src/build_swe.py
fi

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
echo "== guard exit-code contract (what a Muse Code wrapper sees) =="
for pair in "rm -rf /usr:3" "cat ~/.ssh/id_rsa:2" "ls -la:0"; do
  cmd="${pair%:*}"; want="${pair##*:}"
  set +e; "$PY" src/muse_guard_cli.py --no-model "$cmd" >/dev/null 2>&1; got=$?; set -e
  if [ "$got" != "$want" ]; then
    echo "  FAIL '$cmd' exited $got, wanted $want"; exit 1
  fi
  echo "  '$cmd' -> exit $got"
done

echo
echo "ALL CHECKS PASSED"
