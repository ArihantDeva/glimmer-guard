"""Train Llama-3.2-1B as a non-autoregressive tool-failure predictor.

Mechanism: no generation. We read the LM head's logits for the single-token
candidates " yes"/" no" at the final position and softmax over just those two.
p(fail) is that distribution. LoRA keeps optimizer state small enough to fit
one 44GB L40S (full-param AdamW OOMs at this size).

The {allow, ask, block} decision is NOT produced by the model - it is a
threshold on p(fail) applied in code, which is the whole point of a System One
split: the model supplies a probability, the caller decides.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module scope: __file__ is only reliably bound at import time.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("JEV_DATA", os.path.join(ROOT, "data"))


#: Which corpus to train on. The safety corpora and the SWE corpora are
#: different questions ("is this unsafe" vs "will this error") and must never be
#: mixed silently -- that mismatch is what made the first fine-tune score below
#: its floor on both safety benchmarks. Selecting it explicitly keeps the
#: objective visible at the call site.
CORPUS = os.environ.get("JEV_CORPUS", "swe")


def load(split):
    return [json.loads(l) for l in open(f"{D}/{CORPUS}_{split}.jsonl")]


def cand_ids(tok):
    out = {}
    for w in ("yes", "no"):
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"'{w}' is {len(ids)} tokens - masking impossible")
        out[w] = ids[0]
    return out


@torch.no_grad()
def score(model, tok, prompts, cids, bs=16, maxlen=256):
    model.eval()
    probs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        # Only the last position matters, and only two vocabulary entries.
        # Taking the full [B,T,V] logits tensor costs ~13GB here and OOMs.
        out = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        pair = out.logits[torch.arange(out.logits.size(0)), last][
            :, candidates(cids)].float()
        del out
        probs.extend(p_fail_from_pair(pair.cpu().numpy(), cids).tolist())
    return np.array(probs)


# Metrics come from the shared module. A local copy is how the README ends up
# disagreeing with the tests: two implementations, two answers. The trainer
# only trains and scores; every published number is computed by metrics.py.
from noul import candidates, p_fail_from_pair, target_index  # noqa: E402
from train_util import ensure_pad
from metrics import (          # noqa: E402
    apply_temperature as apply_T,
    fit_temperature as fit_T,
    gate as gate_metrics,
    summarize,
)

def report(tag, p, y):
    s = summarize(p, y)
    print(f"  {tag:10s} acc {s['acc']*100:5.1f}%  brier {s['brier']:.4f}  "
          f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
          f"cov@5 {s['cov5']*100:4.0f}%  cov@10 {s['cov10']*100:3.0f}%  "
          f"cov@20 {s['cov20']*100:3.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default=os.environ.get("JEV_OUT", os.path.join(ROOT, "out")))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model)
    ensure_pad(tok)
    cids = cand_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    # LoRA freezes the base embeddings, so checkpointing would otherwise get a
    # graph with no grad_fn. Force the input embeddings to require grad.
    model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    tr, dv, te = load("train"), load("dev"), load("test")
    print(f"seed={a.seed} device={dev} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

    zy = np.array([r["label"] for r in te])
    xt = [r["prompt"] for r in te]
    zp = score(model, tok, xt, cids, maxlen=a.maxlen)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(math.ceil(len(tr) / a.bs * a.epochs))
    model.train()
    t0, gi = time.time(), 0
    for s in range(steps):
        b = [tr[(gi + k) % len(tr)] for k in range(a.bs)]
        gi += a.bs
        enc = tok([r["prompt"] for r in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.maxlen).to(dev)
        out = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        pair = out.logits[torch.arange(len(b)), last][:, candidates(cids)]
        del out
        tgt = torch.tensor([target_index(r["label"], cids) for r in b],
                           device=dev)
        loss = F.cross_entropy(pair.float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 200 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    dp = score(model, tok, [r["prompt"] for r in dv], cids, maxlen=a.maxlen)
    T = fit_T(dp, np.array([r["label"] for r in dv]))
    tp = score(model, tok, xt, cids, maxlen=a.maxlen)
    tpc = apply_T(tp, T)

    print(f"\n=== TEST n={len(te)} fail={int(zy.sum())} T={T:.2f} ===")
    report("zero-shot", zp, zy)
    report("lora", tp, zy)
    report("lora+T", tpc, zy)
    print("\n=== GATE SWEEP on lora+T (block=hi, ask=lo..hi, allow<lo) ===")
    print(f"  {'hi':>4} {'lo':>4} {'block%':>7} {'ask%':>6} {'recall':>7} {'prec':>6} {'falseblk':>9}")
    for hi, lo in ((0.9, 0.7), (0.8, 0.5), (0.7, 0.3), (0.6, 0.2), (0.5, 0.2)):
        g = gate_metrics(tpc, zy, hi, lo)
        print(f"  {hi:4.1f} {lo:4.1f} {g['block_rate']*100:6.1f}% {g['ask_rate']*100:5.1f}% "
              f"{g['fail_recall']*100:6.1f}% {g['block_precision']*100:5.1f}% "
              f"{g['false_block_rate']*100:8.1f}%")
    g = gate_metrics(tpc, zy, 0.7, 0.3)

    res = {"seed": a.seed, "T": T, "n": len(te),
           "zero_shot": {"acc": float(((zp >= .5).astype(int) == zy).mean()),
                         "brier": float(np.mean((zp - zy) ** 2)), "ece": summarize(zp, zy)["ece"]},
           "lora": {"acc": float(((tp >= .5).astype(int) == zy).mean()),
                    "brier": float(np.mean((tp - zy) ** 2)),
                    **{k: v for k, v in summarize(tp, zy).items()
                       if k in ("ece", "aurc", "cov5")}},
           "lora_T": {"acc": float(((tpc >= .5).astype(int) == zy).mean()),
                      "brier": float(np.mean((tpc - zy) ** 2)),
                      **{k: v for k, v in summarize(tpc, zy).items()
                         if k in ("ece", "aurc", "cov5", "cov10", "cov20")}},
           "gate": g}
    with open(f"{a.out}/lora_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/adapter_seed{a.seed}", flush=True)


if __name__ == "__main__":
    main()
