"""Assert that training improved the labelled signal, not just the loss.

Motivation: a masked-token classifier can train to a low cross-entropy while
learning the INVERSE convention (prompt says "yes = fail" but the target for a
failing example is the id of "no"). That run reports a falling loss, saves an
adapter, and predicts backwards. Loss alone is not evidence the model learned
the intended thing, so the checks here are on the deltas that matter:

  * p(fail) must be POSITIVELY correlated with the failure label
  * fine-tuning must not reduce accuracy below the trivial floor
  * temperature scaling must not change accuracy (it is monotone)

Usage: python src/verify_run.py --adapter DIR
"""
import argparse
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_util import ensure_pad
from metrics import apply_temperature, fit_temperature, summarize  # noqa: E402
from train_lora import cand_ids, load, score  # noqa: E402

BASE = "unsloth/Llama-3.2-1B"


def check(adapter):
    tok = AutoTokenizer.from_pretrained(BASE)
    ensure_pad(tok)
    cids = cand_ids(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16).to(dev)
    model = PeftModel.from_pretrained(base, adapter)

    te, dv = load("test"), load("dev")
    y = np.array([r["label"] for r in te])
    dy = np.array([r["label"] for r in dv])

    tp = score(model, tok, [r["prompt"] for r in te], cids)
    with model.disable_adapter():
        zp = score(model, tok, [r["prompt"] for r in te], cids)
    dp = score(model, tok, [r["prompt"] for r in dv], cids)
    T = fit_temperature(dp, dy)
    tpc = apply_temperature(tp, T)

    floor = max(y.mean(), 1 - y.mean())
    z, t, tc = summarize(zp, y), summarize(tp, y), summarize(tpc, y)
    corr = float(np.corrcoef(tp, y)[0, 1])

    print(f"adapter={adapter}  T={T:.2f}  n={len(y)}  floor={floor:.3f}")
    for nm, s in (("zero_shot", z), ("lora", t), ("lora+T", tc)):
        print(f"  {nm:10s} acc {s['acc']*100:5.1f}%  ece {s['ece']:.4f}  "
              f"brier {s['brier']:.4f}  cov@5 {s['cov5']*100:3.0f}%")
    print(f"  corr(p_fail, y) = {corr:+.4f}")

    fails = []
    if corr <= 0:
        fails.append(f"corr {corr:+.4f} is not positive: p(fail) does not track "
                     "failure, so the label convention is inverted or unlearned")
    if t["acc"] + 0.02 < floor:
        fails.append(f"lora acc {t['acc']:.4f} is below the {floor:.4f} floor")
    if t["acc"] < z["acc"]:
        fails.append(f"fine-tuning made accuracy worse ({z['acc']:.4f} -> "
                     f"{t['acc']:.4f})")
    if abs(t["acc"] - tc["acc"]) > 1e-9:
        fails.append("temperature scaling changed accuracy; it must be monotone")
    if tc["ece"] > t["ece"] + 1e-6:
        fails.append(f"temperature made ECE worse ({t['ece']:.4f} -> "
                     f"{tc['ece']:.4f})")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nOK: signal is positive, above floor, and calibration improved.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    sys.exit(check(ap.parse_args().adapter))
