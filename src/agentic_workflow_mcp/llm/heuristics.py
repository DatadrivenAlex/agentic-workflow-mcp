"""Deterministic routing and writing used by the stub and as a fallback."""

from __future__ import annotations

from agentic_workflow_mcp.llm.base import RouteDecision, RouteSnapshot


def heuristic_route(snapshot: RouteSnapshot) -> RouteDecision:
    if snapshot.step >= snapshot.max_steps:
        return RouteDecision(next_agent="finish", reason="step budget exhausted")
    if not snapshot.has_research:
        return RouteDecision(next_agent="researcher", reason="no research notes yet")
    if not snapshot.has_artifact:
        return RouteDecision(next_agent="executor", reason="research is ready for execution")
    return RouteDecision(next_agent="finish", reason="research and artifact are complete")


def build_research(goal: str, evidence: list[str]) -> str:
    blocks = [item.strip() for item in evidence if item and item.strip()]
    body = "\n".join(blocks) if blocks else "No supporting evidence was returned."
    return f"Research for '{goal}':\n{body}"
