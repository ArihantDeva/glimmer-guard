"""Deterministic label rules for tool-call observations.

Design constraint: every function here is a pure function of text. No model,
no randomness, no network. That is what makes the labels auditable and what
lets the whole dataset be rebuilt bit-for-bit from source.

The ordering of `_RULES` matters: the first matching class wins, so the
specific classes are tested before the generic catch-all.
"""
import re

#: The closed set of failure classes the guard predicts. "ok" is the
#: no-failure class, so every observation maps to exactly one member.
ERROR_CLASSES = (
    "ok",
    "import_error",
    "permission_denied",
    "not_found",
    "syntax_error",
    "timeout",
    "runtime_error",
)

#: Ordered (class, pattern) pairs. Specific before generic.
_RULES = (
    ("import_error", re.compile(r"\b(ImportError|ModuleNotFoundError)\b")),
    ("syntax_error", re.compile(r"\bSyntaxError\b|\bIndentationError\b")),
    ("permission_denied", re.compile(r"\bPermission denied\b|\bEACCES\b|\bOperation not permitted\b")),
    ("not_found", re.compile(r"\bcommand not found\b|\bNo such file or directory\b|\bnot found\b")),
    ("timeout", re.compile(r"\btimed out\b|\bTimeout\b|\bTerminated\b|\bCANCELLED\b|\bDeadlineExceeded\b")),
    ("runtime_error", re.compile(r"\b[A-Z]\w*(Error|Exception)\b|\bTraceback \(most recent call last\)")),
)

#: Standalone markers that make an observation an error even when no class
#: rule matched. Anchored so prose containing the word "error" does not fire.
_ERROR_MARKER = re.compile(
    r"^\s*(Traceback \(most recent call last\)|ERROR\b|FATAL\b|error:)"
    r"|\bexit code [1-9]"
    r"|\bexit_code\": [1-9]",
    re.MULTILINE | re.IGNORECASE,
)

#: ``` fenced block, capturing the body.
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def is_error(observation):
    """True when the observation represents a failed tool call.

    An empty or whitespace-only observation means the tool produced no output,
    which is not a failure.
    """
    if not observation or not observation.strip():
        return False
    if _ERROR_MARKER.search(observation):
        return True
    return any(rx.search(observation) for _, rx in _RULES)


def classify_failure(observation):
    """Map an observation to exactly one member of ERROR_CLASSES."""
    if not observation or not observation.strip():
        return "ok"
    for name, rx in _RULES:
        if rx.search(observation):
            return name
    if _ERROR_MARKER.search(observation):
        return "runtime_error"
    return "ok"


def split_command(text):
    """Return the command body from the first fenced block, or None.

    The trajectories put the tool invocation in a fenced block inside the
    assistant message; everything outside the fence is narration.
    """
    if not text:
        return None
    m = _FENCE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


def pair_steps(trajectory):
    """Pair each ai fenced command with the observation that follows it.

    Only ai -> user adjacency produces a pair. A trailing command with no
    following observation is dropped: we never invent an outcome.
    """
    pairs = []
    for i, turn in enumerate(trajectory):
        if turn.get("role") != "ai":
            continue
        command = split_command(turn.get("text"))
        if command is None:
            continue
        if i + 1 >= len(trajectory):
            continue
        nxt = trajectory[i + 1]
        if nxt.get("role") != "user":
            continue
        observation = nxt.get("text") or ""
        pairs.append({
            "command": command,
            "observation": observation,
            "is_error": is_error(observation),
            "failure_class": classify_failure(observation),
        })
    return pairs
