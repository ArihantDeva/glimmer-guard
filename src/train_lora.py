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

D = os.environ.get("JEV_DATA", os.path.expanduser("~/jevlab/data"))


def load(split):
    return [json.loads(l) for l in open(f"{D}/swe_{split}.jsonl")]


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
    order = [cids["yes"], cids["no"]]
    probs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        # Only the last position matters, and only two vocabulary entries.
        # Taking the full [B,T,V] logits tensor costs ~13GB here and OOMs.
        out = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        pair = out.logits[torch.arange(out.logits.size(0)), last][:, order]
        del out
        probs.extend(F.softmax(pair.float(), -1)[:, 1].cpu().tolist())
    return np.array(probs)


def ece(p, y, bins=15):
    conf, pred = np.maximum(p, 1 - p), (p >= .5).astype(int)
    correct = (pred == y).astype(float)
    e = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (conf >= e[i]) & (conf <= e[i + 1] if i == bins - 1 else conf < e[i + 1])
        if m.sum():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def aurc(p, y):
    conf, wrong = np.maximum(p, 1 - p), ((p >= .5).astype(int) != y).astype(float)
    o = np.argsort(-conf)
    return float(np.mean(np.cumsum(wrong[o]) / np.arange(1, len(wrong) + 1)))


def cov_at(p, y, budget):
    conf, wrong = np.maximum(p, 1 - p), ((p >= .5).astype(int) != y).astype(int)
    o, w = np.argsort(-conf), wrong[np.argsort(-conf)]
    best = 0
    for k in range(1, len(w) + 1):
        if w[:k].mean() <= budget:
            best = k
        else:
            break
    return best / len(w)


def fit_T(p, y):
    lo = torch.tensor(np.clip(p, 1e-6, 1 - 1e-6), dtype=torch.float64)
    lg = torch.log(lo) - torch.log1p(-lo)
    yt = torch.tensor(y, dtype=torch.float64)
    best = (1.0, 1e9)
    for t in np.arange(.25, 20.01, .05):
        nll = F.binary_cross_entropy(torch.sigmoid(lg / t).clamp(1e-6, 1 - 1e-6), yt).item()
        if nll < best[1]:
            best = (float(t), nll)
    return best[0]


def apply_T(p, t):
    lo = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(lo) - np.log1p(-lo)) / t))


def gate_metrics(p, y, hi=0.6, lo=0.2):
    """3-way gate from a probability. block where p>=hi, ask in the middle."""
    block, ask = p >= hi, (p >= lo) & (p < hi)
    caught = (block & (y == 1)).sum()
    recall = caught / max((y == 1).sum(), 1)
    false_block = (block & (y == 0)).sum()
    precision = caught / max(block.sum(), 1)
    return {"block_rate": float(block.mean()), "ask_rate": float(ask.mean()),
            "fail_recall": float(recall), "block_precision": float(precision),
            "false_block_rate": float(false_block / max((y == 0).sum(), 1))}


def report(tag, p, y):
    print(f"  {tag:10s} acc {( (p>=.5).astype(int)==y).mean()*100:5.1f}%  "
          f"brier {np.mean((p-y)**2):.4f}  ece {ece(p,y):.4f}  aurc {aurc(p,y):.4f}  "
          f"cov@5 {cov_at(p,y,.05)*100:4.0f}%  cov@10 {cov_at(p,y,.10)*100:3.0f}%  "
          f"cov@20 {cov_at(p,y,.20)*100:3.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("JEV_OUT", os.path.expanduser("~/jevlab/out")))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
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
    order = [cids["yes"], cids["no"]]
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
        pair = out.logits[torch.arange(len(b)), last][:, order]
        del out
        tgt = torch.tensor([r["label"] for r in b], device=dev)
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
                         "brier": float(np.mean((zp - zy) ** 2)), "ece": ece(zp, zy)},
           "lora": {"acc": float(((tp >= .5).astype(int) == zy).mean()),
                    "brier": float(np.mean((tp - zy) ** 2)), "ece": ece(tp, zy),
                    "aurc": aurc(tp, zy), "cov5": cov_at(tp, zy, .05)},
           "lora_T": {"acc": float(((tpc >= .5).astype(int) == zy).mean()),
                      "brier": float(np.mean((tpc - zy) ** 2)), "ece": ece(tpc, zy),
                      "aurc": aurc(tpc, zy), "cov5": cov_at(tpc, zy, .05),
                      "cov10": cov_at(tpc, zy, .10), "cov20": cov_at(tpc, zy, .20)},
           "gate": g}
    with open(f"{a.out}/lora_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/adapter_seed{a.seed}", flush=True)


if __name__ == "__main__":
    main()
