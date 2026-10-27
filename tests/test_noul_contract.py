"""The prompt and the training target must agree on which word means FAIL.

This is the contract that makes a masked-token classifier trustworthy. If the
prompt tells the model "'yes' means it will fail" but the cross-entropy target
for a failing example is the id of "no", the model is trained to contradict
its own instructions. That does not raise, does not look wrong in the logs,
and quietly wastes the fine-tune — so it is pinned here.

The contract has exactly two facts:
  * which word the prompt designates as FAIL
  * which position in the softmax pair carries that word
and they must be the same word.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import noul  # noqa: E402


def test_fail_word_is_declared_and_is_a_candidate():
    assert noul.FAIL_WORD in noul.WORDS
    assert noul.ok_word() in noul.WORDS
    assert noul.FAIL_WORD != noul.ok_word()


def test_instruction_names_the_fail_word():
    """A reader of the prompt must be able to tell which answer means NO.

    The same answer word carries both tasks ("yes" = the bad outcome), so each
    instruction must bind the word to its own question. The failure instruction
    binds it to failing; the safety instruction binds it to being unsafe.
    """
    fail_txt = noul.FAILURE_INSTRUCTION.lower()
    assert noul.FAIL_WORD in fail_txt, (
        f"FAILURE_INSTRUCTION never mentions {noul.FAIL_WORD!r}, so the model "
        "cannot know which word signals failure"
    )
    assert "fail" in fail_txt

    safe_txt = noul.INSTRUCTION.lower()
    assert noul.FAIL_WORD in safe_txt
    assert "unsafe" in safe_txt


def test_fail_index_follows_the_fail_word():
    """With FAIL_WORD='yes', p_fail must be the softmax slot for 'yes'."""
    cids = {"yes": 19690, "no": 916}
    assert noul.fail_index(cids) == 0


def test_the_fail_slot_always_holds_the_fail_token():
    """The invariant that matters, under both conventions.

    The pair is ordered [FAIL, OK], so the *position* of FAIL is 0 either way;
    flipping the convention changes which token id occupies that slot. Asserting
    a shifting position would be testing the wrong thing. What must hold is
    that whatever slot fail_index names is the id of FAIL_WORD.
    """
    cids = {"yes": 19690, "no": 916}
    original = noul.FAIL_WORD
    try:
        for word in ("yes", "no"):
            noul.FAIL_WORD = word
            cand = noul.candidates(cids)
            assert len(set(cand)) == 2, (
                "flipping FAIL_WORD collapsed both softmax slots onto one token"
            )
            assert cand[noul.fail_index(cids)] == cids[word]
            assert cand[noul.fail_index(cids)] != cids[noul.ok_word()]
    finally:
        noul.FAIL_WORD = original


def test_target_index_matches_the_word_the_prompt_names():
    """label==1 means FAIL, so its target must be the FAIL_WORD's slot."""
    cids = {"yes": 19690, "no": 916}
    assert noul.target_index(1, cids) == noul.fail_index(cids)
    assert noul.target_index(0, cids) != noul.fail_index(cids)


def test_probability_is_p_fail_not_p_ok():
    """p_to_fail must return the FAIL_WORD slot, and be 1 - the OK slot."""
    cids = {"yes": 19690, "no": 916}
    import numpy as np
    pair = np.array([[2.0, 0.0]])          # yes much more likely than no
    p = noul.p_fail_from_pair(pair, cids)
    assert p[0] > 0.8, "p_fail should be high when 'yes' dominates"
    assert p[0] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)), rel=1e-6)


def test_ok_probability_is_the_complement():
    import numpy as np
    cids = {"yes": 19690, "no": 916}
    pair = np.array([[2.0, 0.0]])
    assert noul.p_fail_from_pair(pair, cids)[0] + \
        noul.p_ok_from_pair(pair, cids)[0] == pytest.approx(1.0)


def test_label_encoding_is_documented():
    """label==1 must mean 'this call failed' everywhere."""
    assert noul.LABEL_FAIL == 1
    assert noul.LABEL_OK == 0


def test_trainer_uses_the_shared_target_and_read():
    """The trainer must not re-derive the target index or the read slot.

    Re-deriving is precisely how prompt and target drifted apart. Both the
    cross-entropy target and p(fail) must come from noul.
    """
    import re
    trainer = os.path.join(os.path.dirname(__file__), "..", "src", "train_lora.py")
    src = open(trainer).read()
    assert "from noul import" in src or "import noul" in src, (
        "train_lora.py must take the label convention from noul"
    )
    assert "target_index" in src, "trainer must use noul.target_index for its target"
    assert "p_fail_from_pair" in src, "trainer must use noul.p_fail_from_pair to read"


def test_probe_uses_the_shared_read():
    probe = os.path.join(os.path.dirname(__file__), "..", "src", "glimmer_probe.py")
    src = open(probe).read()
    assert "from noul import" in src or "import noul" in src, (
        "glimmer_probe.py must read p(fail) through noul"
    )
    assert "p_fail_from_pair" in src


def test_instruction_is_the_one_noul_ships():
    """The dataset builder must use noul.INSTRUCTION, not a private copy."""
    builder = os.path.join(os.path.dirname(__file__), "..", "src", "build_swe.py")
    src = open(builder).read()
    assert "noul.INSTRUCTION" in src or "from noul import INSTRUCTION" in src, (
        "build_swe.py must build prompts from noul.INSTRUCTION so the prompt "
        "text and the graded word cannot drift"
    )


# ------------------------------------- the instruction must match the corpus

def test_instruction_names_the_question_the_safety_corpus_asks():
    """The first fine-tune asked "will it FAIL" while the safety labels meant
    "is it UNSAFE". Both are single-token decisions, so nothing errored; the
    model simply learned the wrong question and scored below the floor on both
    safety benchmarks. The instruction is part of the label contract."""
    import noul
    txt = noul.INSTRUCTION.lower()
    assert "safe" in txt, f"instruction never mentions safety: {noul.INSTRUCTION!r}"
    assert "fail" not in txt.replace("fail to", ""), (
        "instruction still asks about failure; that is the other task")


def test_safety_prompt_matches_how_training_rows_are_built():
    """Eval prompts and training prompts must be produced by one owner, or the
    model is scored on a format it never saw."""
    import noul
    p = noul.safety_prompt("ctx", ["c1"], "rm -rf /")
    assert "rm -rf /" in p and "c1" in p and "ctx" in p
    assert "safe" in p.lower()
