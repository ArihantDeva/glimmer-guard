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
| zero-shot | 65.2% | 0.2796 | 0.2360 | 0.3750 | 0% |
| + LoRA | 72.2 / 75.0 / 75.3 | 0.205–0.219 | 0.141–0.159 | 0.18–0.28 | 0 / 2 / 7% |
| + temperature | 72.2 / 75.0 / 75.3 | **0.183–0.202** | **0.033 / 0.045 / 0.047** | 0.18–0.28 | 0 / 1 / 7% |

**Majority-class floor: 70.6%.** Read every accuracy above against that number.

What this shows: the mechanism works (single-token masking, no generation),
calibration improves ~5× with a single fitted temperature, and accuracy beats
the floor by 3.6–4.7 points. What it does **not** show: a solved task. See
*Limitations*.

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
scripts/check.sh   the gate behind `make check`
```

## Licence

Code MIT. Model weights inherit from their base models — see
[MODEL_CARD.md](MODEL_CARD.md).
