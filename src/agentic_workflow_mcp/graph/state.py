"""State carried through the compiled graph."""

from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict):
    goal: str
    thread_id: str
    step: int
    max_steps: int
    route: str
    route_history: list[str]
    research_notes: list[str]
    artifacts: list[str]
    tool_trace: list[dict[str, str]]
    trace: list[str]
    final_answer: str
    reason: str
