"""The Noul decision contract: prompt, target, and probability in one place.

A masked-token classifier has three things that must agree, and they live in
three different places unless something forces them together:

  1. the PROMPT, which tells the model which word means failure
  2. the TARGET, which the cross-entropy trains toward for a failing example
  3. the READ, which slot of the two-logit softmax becomes p(fail)

This module exists because they disagreed. The prompt said
"'yes' means it will fail", while `order = [yes, no]` with label 1 trained
toward index 1 -- the id of "no". Training therefore taught the model to
contradict its own instructions: an anti-correlated target that produces no
error, no exception, and plausible-looking loss curves.

Everything downstream (trainer, evaluator, guard) reads the convention from
here so it can only be changed in one place, deliberately.

The decision rule itself is NOT here: the model returns p(fail) and the
caller applies thresholds via metrics.gate(). That separation is the point --
the model supplies evidence, the application owns the policy.
"""

import numpy as np

#: label==1 means "this tool call failed".
LABEL_FAIL = 1
LABEL_OK = 0

#: The two answer words. The only thing a human sets is WORDS; which of them
#: means FAIL is FAIL_WORD, and the other one is derived. Two independent
#: globals let a caller flip one and silently collapse both softmax slots onto
#: the same token -- which is exactly the bug this module exists to prevent.
WORDS = ("yes", "no")

#: The word the prompt designates as FAIL. Everything else follows from this.
#:
#: Chosen empirically, not arbitrarily. Training a failing example toward "yes"
#: reinforces Llama's pre-existing affirmative prior: mean p(fail) rose to 0.79
#: while corr(p, y) stayed at +0.12, i.e. the model got more agreeable instead
#: of more discriminating. Training toward "no" fights that prior and yields
#: corr magnitude ~0.30 -- a far stronger signal. The answer word is arbitrary;
#: what matters is that INSTRUCTION, target_index and p_fail_from_pair name the
#: SAME word. They now agree on "no".
FAIL_WORD = "no"

#: Three-way action labels the application maps probabilities onto.
ACTIONS = ("allow", "ask", "block")

INSTRUCTION = (
    "You are a tool-call guard for a coding agent.\n"
    "Given the command about to run, decide whether it will FAIL.\n"
    "Answer with exactly one word: no (it will fail) or yes (it will succeed)."
)


def ok_word():
    """The other word. Derived, so it can never drift from FAIL_WORD."""
    others = [w for w in WORDS if w != FAIL_WORD]
    if len(others) != 1:
        raise ValueError(f"FAIL_WORD={FAIL_WORD!r} is not one of {WORDS}")
    return others[0]


def candidates(cids):
    """The two vocabulary ids in the order they enter the softmax pair."""
    return [cids[FAIL_WORD], cids[ok_word()]]


def fail_index(cids):
    """Index of FAIL_WORD inside `candidates(cids)`.

    The pair is ordered [FAIL_WORD, ok_word()], so this is always 0. It is
    computed rather than written as 0 so that reordering `candidates` can
    never desynchronise the read from the target.
    """
    return candidates(cids).index(cids[FAIL_WORD])


def target_index(label, cids):
    """Cross-entropy target for `label` (1 = fail)."""
    return fail_index(cids) if int(label) == LABEL_FAIL else 1 - fail_index(cids)


def _softmax2(pair):
    pair = np.asarray(pair, dtype=np.float64)
    e = np.exp(pair - pair.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def p_fail_from_pair(pair, cids):
    """p(fail) from the two logits at the final position."""
    return _softmax2(pair)[:, fail_index(cids)]


def p_ok_from_pair(pair, cids):
    """p(ok), the complement. Exists so a caller cannot silently invert."""
    return _softmax2(pair)[:, 0 if fail_index(cids) == 1 else 1]


def decide(p, hi=0.7, lo=0.3):
    """Map p(fail) to one of ACTIONS. Thresholds live with the caller."""
    if p >= hi:
        return "block"
    if p >= lo:
        return "ask"
    return "allow"
