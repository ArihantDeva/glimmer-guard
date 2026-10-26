# System One Glimmer

> **Not maintained.** Published as a hackathon submission and a record of its
> measurements. No PRs, no issues, no support. See [CONTRIBUTING.md](CONTRIBUTING.md).

A calibrated **non-autoregressive** decision model that predicts whether an
autonomous coding agent's tool call will fail — and returns a probability, not
a sentence.

Built for the **Meta Global AI Developer Hackathon** (indie track), theme
*Push the Frontier*, against Meta's open-weight **Muse Glimmer 30B**, with
**Llama-3.2-1B** as the mechanism rung.

## The idea in one paragraph

Frontier agents spend enormous context deciding small things. A guard has to
answer a much narrower question — *will this command fail?* — millions of
times per day, and it must be both **fast** and **honest about its
uncertainty**. So: no generation. One forward pass, a softmax over `yes`/`no`,
and a threshold in code that maps the probability to allow / ask / block.

## Reproduce everything

```sh
make check          # 43 unit tests + dataset integrity + leak gate + floor test
```

Numbers in this README are produced by that command or by the scripts it
names. Nothing here is a proxy signal.

## Results

### Llama-3.2-1B mechanism rung — clean, leak-free split, 3 seeds

| config | accuracy | Brier | ECE | AURC | cov@5% |
|---|---|---|---|---|---|
| zero-shot | 65.2% | 0.2803 | 0.2371 | 0.3708 | 0% |
| + LoRA | 72.2 / 75.0 / **75.4** | 0.205–0.219 | 0.140–0.161 | 0.177–0.252 | 1 / 2 / 7% |
| + temperature | 72.2 / 75.0 / **75.4** | **0.183–0.202** | **0.033 / 0.044 / 0.046** | 0.177–0.252 | 1 / 2 / 7% |

Per seed, `+temperature`: cov@10% = 17 / 2 / 17, cov@20% = 36 / 4 / 31.
Test split: n = 676, 199 failures (29.4% base rate).

**Majority-class floor: 70.6%.** Read every accuracy above against that number.

What this shows: the mechanism works (single-token masking, no generation),
calibration improves ~5× with a single fitted temperature, and accuracy beats
the floor by 3.6–4.7 points. What it does **not** show: a solved task. See
*Limitations*.

### Muse Glimmer 30B — zero-shot (the honest starting point)

The headline model, asked the task with no fine-tuning, 4-bit NF4, single
forward pass, n=676 test rows, 199 failures:

| config | acc | Brier | ECE | AURC | cov@5% |
|---|---|---|---|---|---|
| Glimmer 30B zero-shot | **44.7%** | 0.3323 | 0.2504 | 0.5611 | 1% |
| always "ok" | 70.6% | — | — | — | — |
| **lift** | **−25.9 pts** | | | | |

`corr(p, y) = +0.050`. AURC 0.5611 is worse than uninformative ranking (0.5).

**Read this plainly: the 30B model, asked directly, is worse than answering
"ok" every time.** Scale does not solve this task. The mechanism — forcing the
answer into two masked tokens and fitting a temperature — is what produces a
usable probability. That is the finding, and it is why the fine-tuned rung
below matters.

### Dataset

| | value |
|---|---|
| raw (command, observation) pairs | 19,210 |
| unique commands after dedup | 6,764 |
| train / dev / test | 5,412 / 676 / 676 |
| split leakage | **none** (enforced by `make check`) |
| failure classes | ok, runtime_error, not_found, syntax_error, import_error |

Labels are deterministic functions of the observation text.

## Limitations (do not quote a number without these)

- **Accuracy is near the floor.** The command text alone under-determines the
  outcome; a large share of failures are contextual.
- **Deduplicate before splitting.** The naive split leaked 606 prompts
  train→test and inflated the score. `make check` now fails on any leak.
- **cov@5% is unstable at n=676.** One row is 0.15 pp. Treat seed-to-seed
  swings (0% → 7%) as noise until the split grows.
- **This is a failure predictor, not a safety judge.** It answers "will this
  error", not "is this harmful".

## Layout

```
src/labels.py      deterministic label rules (24 tests)
src/metrics.py     brier / ece / aurc / coverage_at / temperature / gate (19 tests)
src/build_swe.py   leak-free dataset builder
src/train_lora.py  LoRA + single-token masking trainer
src/rescore.py     re-score a saved adapter (no retraining)
scripts/check.sh   the gate behind `make check`
```

## Licence

Code MIT. Model weights inherit from their base models — see
[MODEL_CARD.md](MODEL_CARD.md).
