"""Composition root: settings, memory, MCP tools, and the compiled graph."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_workflow_mcp.config import Settings
from agentic_workflow_mcp.graph.state import WorkflowState
from agentic_workflow_mcp.graph.workflow import build_graph
from agentic_workflow_mcp.llm.base import LLM
from agentic_workflow_mcp.llm.factory import build_llm
from agentic_workflow_mcp.mcp.server import InProcessMCPServer
from agentic_workflow_mcp.memory.store import LongTermMemory, ShortTermMemory


class RunReport(BaseModel):
    goal: str
    thread_id: str
    provider: str
    final_answer: str
    route_history: list[str]
    trace: list[str]
    tool_trace: list[dict[str, str]] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)


class WorkflowRuntime:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm or build_llm(self.settings)
        self.long_term = LongTermMemory(self.settings.memory_dir / "long_term.sqlite")
        self.short_term = ShortTermMemory(
            window=self.settings.short_term_window,
            path=self.settings.memory_dir / "short_term.json",
        )
        self.mcp = InProcessMCPServer(self.long_term)
        self.graph = build_graph(self.llm, self.mcp, self.short_term)

    def run(self, goal: str, thread_id: str | None = None) -> RunReport:
        text = goal.strip()
        if not text:
            raise ValueError("goal is empty")
        thread = thread_id.strip() if thread_id else f"thread-{uuid4().hex[:8]}"
        if not thread:
            raise ValueError("thread id is empty")
        initial: WorkflowState = {
            "goal": text,
            "thread_id": thread,
            "step": 0,
            "max_steps": self.settings.max_steps,
            "route": "",
            "route_history": [],
            "research_notes": [],
            "artifacts": [],
            "tool_trace": [],
            "trace": [],
            "final_answer": "",
            "reason": "",
        }
        result = self.graph.invoke(initial)
        return RunReport(
            goal=text,
            thread_id=thread,
            provider=self.llm.name,
            final_answer=str(result["final_answer"]),
            route_history=list(result["route_history"]),
            trace=list(result["trace"]),
            tool_trace=list(result["tool_trace"]),
            research_notes=list(result["research_notes"]),
        )
