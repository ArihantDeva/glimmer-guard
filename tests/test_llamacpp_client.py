"""The llama.cpp backend must derive the same probability as transformers.

Two scoring paths exist: in-process transformers (used for evaluation) and the
llama-server HTTP API (used by Muse Code). If they disagree, the published
numbers describe a model nobody is serving. These tests pin the arithmetic
without a server, by injecting a fake transport.

They also pin the property that makes the HTTP path trustworthy: when the server
does not return the two candidate log-probabilities, the client RAISES instead
of defaulting to 0.5. A silent default would look calibrated and carry no
information -- the worst possible failure for a guard.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llamacpp_client  # noqa: E402


def _fake(logprobs_pair, captured=None):
    """A transport that returns a llama.cpp-shaped completion response."""
    def post(url, payload):
        if captured is not None:
            captured.append((url, payload))
        return {"choices": [{"logprobs": {"top_logprobs": [logprobs_pair]}}]}
    return post


# --------------------------------------------------------------- payload

def test_requests_exactly_one_token():
    """max_tokens=1 is a correctness requirement, not an optimisation."""
    seen = []
    s = llamacpp_client.LlamaCppScorer(post=_fake({"yes": -0.1, "no": -2.0}, seen))
    s.p_fail("prompt")
    _, payload = seen[0]
    assert payload["max_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["temperature"] == 0.0


def test_posts_to_the_completions_route():
    seen = []
    s = llamacpp_client.LlamaCppScorer(base_url="http://h:8080/",
                                      post=_fake({"yes": -1, "no": -1}, seen))
    s.p_fail("p")
    assert seen[0][0] == "http://h:8080/v1/completions"


def test_sends_the_prompt_unchanged():
    seen = []
    s = llamacpp_client.LlamaCppScorer(post=_fake({"yes": -1, "no": -1}, seen))
    s.p_fail("some prompt text")
    assert seen[0][1]["prompt"] == "some prompt text"


# ---------------------------------------------------------- arithmetic

def test_equal_logprobs_give_half():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -1.0, "yes": -1.0}))
    assert s.p_fail("p") == pytest.approx(0.5)


def test_fail_word_dominating_gives_high_probability():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -0.1, "yes": -3.0}))
    assert s.p_fail("p") > 0.9


def test_ok_word_dominating_gives_low_probability():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -3.0, "yes": -0.1}))
    assert s.p_fail("p") < 0.1


def test_matches_a_hand_computed_logistic():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -0.5, "yes": -1.5}))
    expected = 1.0 / (1.0 + math.exp(-1.5 - (-0.5)))
    assert s.p_fail("p") == pytest.approx(expected, rel=1e-9)


def test_leading_space_and_case_are_tolerated():
    """llama.cpp reports tokens as they appear in the vocab (' no', 'No')."""
    s = llamacpp_client.LlamaCppScorer(post=_fake({" no": -0.2, " yes": -2.2}))
    assert s.p_fail("p") > 0.8


def test_fail_word_is_configurable():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -3.0, "yes": -0.1}),
                                       fail_word="yes")
    assert s.p_fail("p") > 0.9


# ----------------------------------------------------- failure is loud

def test_missing_both_tokens_raises():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"maybe": -0.1, "ok": -2.0}))
    with pytest.raises(RuntimeError, match="both answer tokens"):
        s.p_fail("p")


def test_missing_one_token_raises_rather_than_defaulting():
    s = llamacpp_client.LlamaCppScorer(post=_fake({"no": -0.1}))
    with pytest.raises(RuntimeError):
        s.p_fail("p")


def test_server_without_logprobs_raises():
    """A 0.5 default would look calibrated while carrying no information."""
    s = llamacpp_client.LlamaCppScorer(post=lambda u, p: {"choices": [{}]})
    with pytest.raises(RuntimeError, match="top_logprobs"):
        s.p_fail("p")


def test_error_message_names_the_transport_not_the_model():
    s = llamacpp_client.LlamaCppScorer(post=lambda u, p: {"choices": []})
    with pytest.raises(RuntimeError, match="llama-server"):
        s.p_fail("p")


# ----------------------------------------------------------- round trip

def test_probability_is_in_range_for_extreme_logprobs():
    for pair in ({"no": -900.0, "yes": -0.0}, {"no": 0.0, "yes": -900.0},
                 {"no": -50.0, "yes": -50.0}):
        s = llamacpp_client.LlamaCppScorer(post=_fake(pair))
        assert 0.0 <= s.p_fail("p") <= 1.0
