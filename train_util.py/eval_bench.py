"""Score a saved adapter on R-Judge and Agent-SafetyBench.

Reports the three configurations the goal requires (zero-shot, fine-tuned,
fine-tuned + temperature) on each benchmark, with the trivial-baseline floor
printed beside every number.

Two honesty rules are enforced in the output rather than in prose:

  * the floor is always printed, because R-Judge is separable by surface
    statistics (+18.5 pts at action level) and a bare accuracy there would be
    misleading
  * the temperature is fitted on a held-out split and reported on the other, so
    the calibrated number is not an in-sample optimum

    python -u src/eval_bench.py --adapter out/adapter_seed0 --bench asb
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evals  # noqa: E402
from metrics import apply_temperature, fit_temperature, gate, summarize  # noqa: E402
from train_lora import cand_ids, score  # noqa: E402

BASE = "unsloth/Llama-3.2-1B"
MAXLEN = 192


def floor(rows):
    y = np.array([r["label"] for r in rows])
    return float(max(y.mean(), 1 - y.mean()))


def run(adapter, bench, maxlen=MAXLEN):
    rows = evals.load_bench(bench)
    # Split so the temperature is fitted out-of-sample. The FULL set is what
    # gets reported, with T from the other half.
    a, b = evals.split_rows(rows, 0.5, seed=0)
    fit_rows, rep_rows = (a, b) if len(a) >= len(b) else (b, a)

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    cids = cand_ids(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16).to(dev)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base

    yv = np.array([r["label"] for r in fit_rows])
    yr = np.array([r["label"] for r in rep_rows])
    pv = score(model, tok, [r["prompt"] for r in fit_rows], cids, maxlen=maxlen)
    tp = score(model, tok, [r["prompt"] for r in rep_rows], cids, maxlen=maxlen)

    # The zero-shot baseline must come from the BASE weights. With an adapter
    # attached that means explicitly switching it off; scoring it while the
    # adapter is live would silently report the fine-tuned model as "baseline".
    if adapter:
        with model.disable_adapter():
            zp = score(model, tok, [r["prompt"] for r in rep_rows], cids,
                       maxlen=maxlen)
    else:
        zp = tp

    T = fit_temperature(pv, yv) if adapter else 1.0
    tpc = apply_temperature(tp, T)

    fl = floor(rows)
    print(f"\n=== {bench.upper()}  n_report={len(rep_rows)} "
          f"fail={int(yr.sum())}  T={T:.2f} ===")
    print(f"  floor (majority class)   {fl*100:5.1f}%   "
          f"<- a constant predictor scores this")
    out = {"bench": bench, "adapter": adapter, "T": T,
           "n_report": len(rep_rows), "floor": fl}
    for tag, p in (("zero-shot", zp), ("finetuned", tp), ("finetuned+T", tpc)):
        s = summarize(p, yr)
        s["corr"] = float(np.corrcoef(p, yr)[0, 1]) if p.std() > 0 else 0.0
        s["gate_0.7_0.3"] = gate(p, yr, 0.7, 0.3)
        out[tag] = s
        print(f"  {tag:12s} acc {s['acc']*100:5.1f}%  brier {s['brier']:.4f}  "
              f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
              f"cov@5 {s['cov5']*100:3.0f}%  "
              f"lift {(s['acc']-fl)*100:+5.1f}pt  corr {s['corr']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bench", default="both",
                    choices=("rjudge", "asb", "both"))
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    benches = ("rjudge", "asb") if a.bench == "both" else (a.bench,)
    res = [run(a.adapter, b, a.maxlen) for b in benches]
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
