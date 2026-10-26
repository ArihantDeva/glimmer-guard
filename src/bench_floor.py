#!/usr/bin/env python3
"""Report the trivial-baseline floor for every benchmark.

A benchmark that a constant or length-only predictor can match cannot support a
claim about model quality. This runs the same floor test used on the training
data against R-Judge and Agent-SafetyBench and prints a verdict per benchmark.

It is part of `make check` because the failure mode is silent: a benchmark with
a surface artefact looks like a real evaluation until someone measures it.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import evals  # noqa: E402

PATHS = {"rjudge": "data/rjudge.jsonl", "asb": "data/asb.json"}


def floor_for(bench):
    rows = evals.load_bench(bench)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        tmp = f.name
    try:
        src = os.path.join(os.path.dirname(__file__), "..", "src", "floor.py")
        out = subprocess.run([sys.executable, src, tmp, tmp],
                             capture_output=True, text=True).stdout
    finally:
        os.unlink(tmp)
    surface = [l.strip() for l in out.splitlines() if "surface stats" in l]
    majority = [l.strip() for l in out.splitlines() if "majority-class" in l]
    return ("FAIL" if "FAIL:" in out else "OK"), surface, majority, out


def main():
    bad = 0
    for bench, path in PATHS.items():
        if not os.path.exists(path):
            print(f"  {bench:8s} SKIP (missing {path} - see src/fetch_rjudge.py)")
            continue
        verdict, surface, majority, out = floor_for(bench)
        n = len(evals.load_bench(bench))
        print(f"  {bench:8s} n={n:5d} {verdict:4s} "
              f"{surface[0] if surface else '(no reading)'}")
        if verdict == "FAIL":
            bad += 1
            print(f"           ^ a trivial predictor matches this benchmark; "
                  f"report it as eval-only, never as a training signal")
    print()
    if bad:
        print(f"FLOOR WARNING: {bad} benchmark(s) are separable by surface form.")
        print("They may still be reported, but the floor MUST appear next to "
              "every number taken from them.")
    else:
        print("OK: no benchmark is separable by a trivial predictor.")


if __name__ == "__main__":
    main()
