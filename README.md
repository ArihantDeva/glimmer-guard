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

Test set n=676, 199 failures (29.4%). **Majority-class floor: 70.6%.**

| config | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| zero-shot accuracy | 64.9% | 64.9% | 64.9% | 64.9% |
| LoRA accuracy | 75.4% | 75.6% | 74.0% | **75.0%** |
| +temperature accuracy | 75.4% | 75.6% | 74.0% | 75.0% |

Calibration and selective prediction:

| metric | zero-shot | +LoRA | +LoRA+temp |
|---|---|---|---|
| Brier | 0.2800 | 0.2029–0.2079 | **0.1858–0.1884** |
| ECE | 0.2426 | 0.1479–0.1501 | **0.0487 / 0.0511 / 0.0517** |
| AURC | 0.3621 | 0.207–0.210 | 0.207–0.210 |
| cov@5% | 0% | 4–7% | 4–7% |

Zero-shot is identical across seeds because scoring the frozen base model is
deterministic.

What this shows: the mechanism works (single-token masking, no generation);
one fitted temperature cuts ECE **~4.8×** (0.2426 → ~0.050); accuracy lands
**+4.4 points above the floor** with a spread of 1.6 points across seeds.
What it does not show: a solved task. See *Limitations*.

### Benchmarks (R-Judge, Agent-SafetyBench)

Method that matters for reading these numbers:

- **Temperature is fitted on the model's own dev split, never on the benchmark.**
  Fitting T on the benchmark would leak it into the calibration curve; fitting
  on half and reporting the other half is honest but wastes half the
  evaluation. The dev split exists to be spent on this.
- **The benchmark is reported in full**, so `n` is the whole set, not a half.
- **The majority-class floor is printed beside every row.** A benchmark a
  constant predictor can match cannot support a claim about a model.

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
| raw (command, observation) pairs | 41,333 |
| unique commands after dedup | 12,744 |
| collapsed as exact duplicates | 28,589 |
| genuine label conflicts | 3,340 |
| train / dev / test | 10,196 / 1,274 / 1,274 |
| distinct SWE-bench instances | 74 |
| split leakage | **none** (enforced by `make check`) |
| failure classes | ok, runtime_error, not_found, syntax_error, import_error |

Labels are deterministic functions of the observation text.

**On the 223k target in the brief: this dataset is 12,744 unique commands, 5.7%
of it.** That is the real number and it is not padded. The raw pair count is
larger only because agents repeat commands; the duplicate and conflict counts
are published so the gap is auditable rather than hidden. Two limits worth
stating plainly:

- **74 source repositories.** Every row comes from an agent working in one of
  74 SWE-bench instances, so the command distribution is narrow: mostly Python
  test runners, file reads, and package installs. 12,744 rows from 74 repos is
  not 12,744 rows from 12,744 repos, and the model's transfer to an unseen
  domain is bounded by that, not by the row count.
- **3,340 label conflicts** are commands that both succeeded and failed in the
  same repository. Those are irreducible from the observation alone: the outcome
  depends on state the prompt does not carry. They are kept, not dropped,
  because removing them would make the task look cleaner than it is.

The R-Judge result below is the same limit measured on someone else's benchmark.

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
