"""Shared LLM types."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class RouteSnapshot(BaseModel):
    goal: str
    step: int
    max_steps: int
    has_research: bool
    has_artifact: bool


class RouteDecision(BaseModel):
    next_agent: str
    reason: str


class LLM(Protocol):
    """Small surface used by the graph and the crew layer."""

    name: str

    def decide_route(self, snapshot: RouteSnapshot) -> RouteDecision:
        """Propose the next specialist. The graph still constrains the route."""

    def research(self, goal: str, evidence: list[str]) -> str:
        """Turn tool evidence into a research note."""

    def execute(self, goal: str, research: str, draft: str) -> str:
        """Turn a drafted brief into the artifact that is stored."""

    def complete(self, system: str, user: str) -> str:
        """Generic completion used by sequential crew tasks."""
