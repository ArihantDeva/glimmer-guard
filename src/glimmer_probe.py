"""Probe Muse Glimmer 30B for single-token yes/no masking.

The mechanism must be proven on the headline model before any fine-tuning:
one forward pass, read two logits at the final position, softmax them. If the
model cannot expose that, the whole design is wrong and no amount of training
will fix it.

This script answers three things:
  1. does transformers 5.x load the muse_glimmer architecture at all
  2. are ' yes'/' no' single tokens in its tokenizer (cand_ids enforces this)
  3. is the resulting p(fail) anti-correlated with the truth (i.e. is the
     model doing anything at all, or is it inverted)

Run one copy per GPU group; it deliberately loads with device_map=auto so it
can span all 8 A100s.
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lora import cand_ids, load  # noqa: E402

DEFAULT = "meta-models/Muse-Glimmer-30B"


def _quant(load_4bit):
    if not load_4bit:
        return {}
    from transformers import BitsAndBytesConfig
    return dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ))


def load_model(name, dev, load_4bit=False):
    """Load Glimmer through whichever auto-class accepts it.

    MuseGlimmer is a multimodal architecture (MuseGlimmerForConditionalGeneration),
    so it is NOT in AutoModelForCausalLM's registry - that raises a 200-entry
    ValueError. Try the multimodal class first, then the plain one, then bare
    AutoModel, and report which one worked.
    """
    import transformers
    kw = dict(device_map=dev, trust_remote_code=True, **_quant(load_4bit))
    if not load_4bit:
        kw["torch_dtype"] = torch.bfloat16

    candidates = [
        ("image-text", getattr(transformers, "AutoModelForImageTextToText", None)),
        ("vision2seq", getattr(transformers, "AutoModelForVision2Seq", None)),
        ("causal-lm", transformers.AutoModelForCausalLM),
        ("auto", transformers.AutoModel),
    ]
    last = None
    for label, cls in candidates:
        if cls is None:
            continue
        try:
            return cls.from_pretrained(name, **kw), label
        except Exception as e:
            last = e
            print(f"  {label} failed: {str(e)[:120]}", flush=True)
    raise SystemExit(f"no auto-class loaded {name}: {last}")


@torch.no_grad()
def p_fail(model, tok, prompts, cids, bs=4, maxlen=192, softcap=None):
    """Return p(fail) for each prompt from a single forward pass each."""
    order = [cids["yes"], cids["no"]]
    out_p = []
    model.eval()
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        pair = logits[torch.arange(logits.size(0)), last][:, order].float()
        del logits
        if softcap:                       # final_logit_softcapping is a tanh
            pair = torch.tanh(pair / softcap) * softcap
        out_p.extend(torch.softmax(pair, -1)[:, 1].cpu().tolist())
    return np.array(out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--softcap", type=float, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump", default=None, help="write npz of p and y")
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cids = cand_ids(tok)          # raises SystemExit if either word is not ONE token
    print(f"tokenizer ok: yes={cids['yes']} no={cids['no']}", flush=True)

    rows = load(a.split)[:a.n]
    y = np.array([r["label"] for r in rows])
    print(f"n={len(rows)} fail={int(y.sum())} base_rate={y.mean():.3f}", flush=True)

    model, kind = load_model(a.model, a.device_map, a.load_4bit)
    print(f"loaded as {kind}", flush=True)

    p = p_fail(model, tok, [r["prompt"] for r in rows], cids,
               bs=a.bs, maxlen=a.maxlen, softcap=a.softcap)

    from metrics import summarize
    s = summarize(p, y)
    corr = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else float("nan")

    # Trivial baselines. A model that always answers the same word is not a
    # classifier, so these belong next to every headline number.
    all_fail = float(y.mean())
    all_ok = float(1 - y.mean())
    majority = max(all_fail, all_ok)

    print(f"\nmean p(fail) = {p.mean():.4f}   mean y = {y.mean():.4f}")
    print(f"p range      = {p.min():.3f} .. {p.max():.3f}")
    print(f"corr(p,y)    = {corr:+.4f}")
    print(f"\n{'config':12s} {'acc':>7} {'brier':>8} {'ece':>8} {'aurc':>8} "
          f"{'cov@5':>7} {'cov@10':>7} {'cov@20':>7}")
    print(f"{'glimmer 0s':12s} {s['acc']*100:6.1f}% {s['brier']:8.4f} {s['ece']:8.4f} "
          f"{s['aurc']:8.4f} {s['cov5']*100:6.0f}% {s['cov10']*100:6.0f}% "
          f"{s['cov20']*100:6.0f}%")
    print(f"{'all-fail':12s} {all_fail*100:6.1f}%  <-- trivial baseline")
    print(f"{'all-ok':12s} {all_ok*100:6.1f}%")
    print(f"{'majority':12s} {majority*100:6.1f}%  <-- floor")
    print(f"\nlift over floor = {(s['acc']-majority)*100:+.1f} pts")
    print(f"first 10 p   = {np.round(p[:10], 3)}")
    print(f"first 10 y   = {y[:10]}")

    if a.dump:
        np.savez(a.dump, p=p, y=y)
        print(f"wrote {a.dump}")


if __name__ == "__main__":
    main()
