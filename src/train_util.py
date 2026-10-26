"""Scheduling helpers that must be testable without a GPU or torch installed.

These live apart from the trainer deliberately. `train_glimmer.py` imports torch
at module scope, so anything defined there cannot be unit-tested on a machine
without torch -- which is every machine except the cluster. A scheduling rule is
pure arithmetic and has no business requiring a 2GB dependency to verify.
"""


def checkpoint_due(step, steps, every):
    """True when this step should write a checkpoint.

    The last step always counts: a run that reaches the end must save, even when
    `every` does not divide `steps`. This exists because the first Glimmer run
    saved ONLY at the end while projecting 3.82 hours against a 4-hour wall
    clock, so a timeout would have destroyed hours of GPU with nothing kept.
    """
    if every <= 0:
        raise ValueError("checkpoint interval must be positive")
    return step % every == 0 or step == steps


def steps_for(n_examples, batch_size, accum, epochs):
    """Gradient steps needed to cover the data once per epoch."""
    if batch_size <= 0 or accum <= 0:
        raise ValueError("batch size and accumulation must be positive")
    import math
    return int(math.ceil(n_examples / (batch_size * accum) * epochs))
