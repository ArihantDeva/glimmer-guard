"""Merge a LoRA adapter into the base weights, for GGUF conversion.

GGUF conversion reads a plain HF checkpoint, so the adapter has to be folded in
first. Merging is lossless here in the sense that matters: the merged model
computes the same function as base+adapter, so the numbers measured through the
transformers path carry over to the served path. If that were not true the GGUF
would be a different model than the one benchmarked.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Llama-3.2-1B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.adapter)

    merged = model.merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(f"merged -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
