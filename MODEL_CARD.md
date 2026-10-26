# Model card — tool-failure Noul head

## What this is

A **non-autoregressive decision model**: given the text of a tool call, it
returns `p(fail)` as a probability. No tokens are generated. The decision is a
single softmax over two vocabulary entries (`" yes"`, `" no"`) read from the
LM head's logits at the final position.

The caller turns `p(fail)` into a three-way action in code, not in the model:

| condition | action |
|---|---|
| `p < lo` | **allow** — run unattended |
| `lo <= p < hi` | **ask** — surface to the human |
| `p >= hi` | **block** — refuse to run |

Thresholds live in the application. The model supplies only a probability.

## Base models and licence obligations

| Model | Licence | Obligation |
|---|---|---|
| `meta-llama/Llama-3.2-1B` | **Llama 3.2 Community License** | §1.b.i: any model created by fine-tuning it **must be named with "Llama" at the front**. Ship this licence. 700M-MAU clause applies. |
| `meta-models/Muse-Glimmer-30B-GGUF` | Meta Glimmer licence (permissive, commercial use) | ship the model card with derivatives |

**This repository's own code is MIT** (`LICENSE`). Model weights are *not* MIT
— they inherit from the base model above. Any published adapter stays
"Llama"-prefixed.

## Intended use

Guarding tool calls from an autonomous coding agent (developed against Muse
Code). The model is trained to predict *failure*, not harmfulness: it says
whether a command will error, not whether it is safe.

## Out of scope

- Adjudicating prompt injection or adversarial intent (see R-Judge limits below)
- Any decision where the cost of a false *allow* is unbounded
- Assuming the model can see paths, permissions, or filesystem state — that is
  the caller's job (`precheck()` in code)

## Training data

Derived from public agent trajectories (`nebius/SWE-agent-trajectories`, MIT).
Labels are **deterministic functions of the observation text** (`src/labels.py`,
24 unit tests) — no human annotation and no LLM in the labelling loop. The
dataset is rebuilt from source, never committed.

## Known limitations — read before quoting a number

1. **Accuracy sits near the majority-class floor.** The task is imbalanced and
   the command text alone under-determines the outcome. Report the floor
   alongside every accuracy figure.
2. **Duplicates collapse hard.** Raw pairs far outnumber unique commands, so
   deduplicate before splitting or the test score is inflated by leakage.
3. **cov@5% is unstable at small n.** On a few-hundred-row test split, one row
   moves coverage by 0.15 pp. Use a CI or a larger split.
4. **R-Judge is not a training set here.** Its whole-interaction labels are
   confounded with interaction length; see `docs/` for the measurement.
