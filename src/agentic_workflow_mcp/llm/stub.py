"""Offline LLM. Routing follows workflow state, not instructions inside the goal."""

from __future__ import annotations

from agentic_workflow_mcp.llm.base import RouteDecision, RouteSnapshot
from agentic_workflow_mcp.llm.heuristics import build_research, heuristic_route


class StubLLM:
    name = "stub"

    def decide_route(self, snapshot: RouteSnapshot) -> RouteDecision:
        return heuristic_route(snapshot)

    def research(self, goal: str, evidence: list[str]) -> str:
        return build_research(goal, evidence)

    def execute(self, goal: str, research: str, draft: str) -> str:
        del goal, research
        cleaned = draft.strip()
        if cleaned:
            return cleaned
        return "No draft was produced."

    def complete(self, system: str, user: str) -> str:
        role = "agent"
        for line in system.splitlines():
            if line.startswith("Role: "):
                role = line.removeprefix("Role: ").strip() or role
                break
        summary = user.strip().splitlines()[0][:240] if user.strip() else "empty task"
        if "Context from earlier tasks:" in user:
            summary = f"{summary}. Used the previous task output."
        return f"{role} completed the task. {summary}"
