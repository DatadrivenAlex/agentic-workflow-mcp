"""Lifecycle guard around whatever route the model proposes."""

from __future__ import annotations

_LEGAL = {"researcher", "executor", "finish"}


def constrain_route(
    proposed: str,
    *,
    step: int,
    max_steps: int,
    has_research: bool,
    has_artifact: bool,
) -> str:
    """Finish only when an artifact exists or the step budget is spent.

    A model that says "finish" on the first turn, or that repeats one specialist,
    is pulled back onto researcher -> executor -> finish.
    """

    if step >= max_steps:
        return "finish"
    if proposed not in _LEGAL:
        proposed = "researcher"
    if not has_research:
        return "researcher"
    if not has_artifact:
        return "executor"
    return "finish"
