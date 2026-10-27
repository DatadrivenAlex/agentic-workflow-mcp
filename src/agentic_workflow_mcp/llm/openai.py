"""Optional OpenAI chat-completions client. Failures fall back to the stub."""

from __future__ import annotations

import json
import logging
import urllib.request

from agentic_workflow_mcp.llm.base import RouteDecision, RouteSnapshot
from agentic_workflow_mcp.llm.heuristics import heuristic_route
from agentic_workflow_mcp.llm.jsonutil import extract_json_object
from agentic_workflow_mcp.llm.stub import StubLLM

logger = logging.getLogger(__name__)

_ALLOWED_AGENTS = {"researcher", "executor", "finish"}


class OpenAILLM:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fallback = StubLLM()

    def decide_route(self, snapshot: RouteSnapshot) -> RouteDecision:
        prompt = (
            "Choose the next agent. Reply with JSON only: "
            '{"next_agent": "researcher" | "executor" | "finish", "reason": "short reason"}.\n'
            f"State: {snapshot.model_dump_json()}"
        )
        try:
            raw = self._chat("You route a research workflow. Do not call tools yourself.", prompt)
            decision = RouteDecision.model_validate(extract_json_object(raw))
        except Exception:
            logger.warning("openai route parse failed; using the offline heuristic")
            return heuristic_route(snapshot)
        if decision.next_agent not in _ALLOWED_AGENTS:
            logger.warning("openai route %s is unknown; using the offline heuristic", decision.next_agent)
            return heuristic_route(snapshot)
        return decision

    def research(self, goal: str, evidence: list[str]) -> str:
        try:
            return self._chat(
                "Write a short research note using only the evidence. Do not invent sources.",
                f"Goal: {goal}\nEvidence:\n" + "\n".join(evidence),
            ).strip()
        except Exception:
            logger.warning("openai research call failed; using the offline stub")
            return self._fallback.research(goal, evidence)

    def execute(self, goal: str, research: str, draft: str) -> str:
        try:
            text = self._chat(
                "Return the final markdown brief. Keep the goal and the findings.",
                f"Goal: {goal}\nResearch:\n{research}\n\nDraft:\n{draft}",
            ).strip()
            return text or draft
        except Exception:
            logger.warning("openai execute call failed; using the offline stub")
            return self._fallback.execute(goal, research, draft)

    def complete(self, system: str, user: str) -> str:
        try:
            return self._chat(system, user).strip()
        except Exception:
            logger.warning("openai completion failed; using the offline stub")
            return self._fallback.complete(system, user)

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
