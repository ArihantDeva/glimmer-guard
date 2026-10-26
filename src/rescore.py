"""Re-score saved adapters through the shared metrics module.

This file deliberately mirrors the scoring path that reproduced 75.30% in
diagnostics: same model loading (no device_map), same tokenizer, same maxlen,
same single-token pair. The only addition is that every metric comes from
metrics.py, so the published numbers and the unit tests cannot diverge.

    python -u src/rescore.py --adapter <dir> [--out <json>]
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
from metrics import (  # noqa: E402
    apply_temperature,
    fit_temperature,
    gate,
    summarize,
)
from train_lora import cand_ids, load, score  # noqa: E402

BASE = "unsloth/Llama-3.2-1B"
MAXLEN = 192          # must equal train_lora.py --maxlen


def _tok():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    return tok


def _score_split(model, tok, cids, rows):
    return score(model, tok, [r["prompt"] for r in rows], cids, maxlen=MAXLEN)


def evaluate(adapter, out=None):
    tok = _tok()
    cids = cand_ids(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16).to(dev)
    model = PeftModel.from_pretrained(base, adapter)

    te, dv = load("test"), load("dev")
    y = np.array([r["label"] for r in te])
    dy = np.array([r["label"] for r in dv])

    tp = _score_split(model, tok, cids, te)
    with model.disable_adapter():
        zp = _score_split(model, tok, cids, te)
    dp = _score_split(model, tok, cids, dv)
    T = fit_temperature(dp, dy)
    tpc = apply_temperature(tp, T)

    print(f"adapter={adapter}", flush=True)
    print(f"n={len(y)} fail={int(y.sum())} base_rate={y.mean():.3f} "
          f"T={T:.2f}", flush=True)

    res = {"adapter": adapter, "T": T, "n": int(len(y))}
    for name, p in (("zero_shot", zp), ("lora", tp), ("lora_T", tpc)):
        s = summarize(p, y)
        s["corr"] = float(np.corrcoef(p, y)[0, 1])
        s["gate_0.7_0.3"] = gate(p, y, 0.7, 0.3)
        res[name] = s
        print(f"  {name:10s} acc {s['acc'] * 100:5.1f}%  brier {s['brier']:.4f}  "
              f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
              f"cov@5 {s['cov5'] * 100:3.0f}%  cov@10 {s['cov10'] * 100:3.0f}%  "
              f"cov@20 {s['cov20'] * 100:3.0f}%", flush=True)

    dest = out or os.path.join(adapter, "rescore.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {dest}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    evaluate(a.adapter, a.out)


if __name__ == "__main__":
    main()
