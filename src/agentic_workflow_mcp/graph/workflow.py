"""Compile the supervisor graph. Nodes call the in-process MCP server."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agentic_workflow_mcp.graph.routing import constrain_route
from agentic_workflow_mcp.graph.state import WorkflowState
from agentic_workflow_mcp.llm.base import LLM, RouteSnapshot
from agentic_workflow_mcp.mcp.server import InProcessMCPServer
from agentic_workflow_mcp.memory.store import ShortTermMemory

logger = logging.getLogger(__name__)


def build_graph(llm: LLM, mcp: InProcessMCPServer, short_term: ShortTermMemory):
    def supervisor(state: WorkflowState) -> dict:
        snapshot = RouteSnapshot(
            goal=state["goal"],
            step=state["step"],
            max_steps=state["max_steps"],
            has_research=bool(state["research_notes"]),
            has_artifact=bool(state["artifacts"]),
        )
        decision = llm.decide_route(snapshot)
        if state["step"] >= state["max_steps"]:
            route = "finish"
            reason = "step budget exhausted"
        else:
            route = constrain_route(
                decision.next_agent,
                step=state["step"],
                max_steps=state["max_steps"],
                has_research=snapshot.has_research,
                has_artifact=snapshot.has_artifact,
            )
            reason = decision.reason
            if route != decision.next_agent:
                reason = f"{reason} (adjusted {decision.next_agent} -> {route})"
        line = f"supervisor -> {route}: {reason}"
        logger.info(
            "route next=%s reason=%s step=%s thread=%s",
            route,
            reason,
            state["step"],
            state["thread_id"],
        )
        short_term.add(state["thread_id"], "supervisor", line)
        update: dict = {
            "route": route,
            "reason": reason,
            "route_history": [*state["route_history"], route],
            "trace": [*state["trace"], line],
        }
        if route == "finish":
            update["final_answer"] = _final_answer(state)
        else:
            update["step"] = state["step"] + 1
        return update

    def researcher(state: WorkflowState) -> dict:
        search = mcp.call_tool("knowledge_search", {"query": state["goal"]})
        recall = mcp.call_tool("memory_recall", {"query": state["goal"]})
        evidence = [search.text]
        if recall.text and recall.text != "No long-term memories matched.":
            evidence.append("Prior memory:\n" + recall.text)
        note = llm.research(state["goal"], evidence)
        line = f"researcher recorded notes ({len(note)} chars)"
        short_term.add(state["thread_id"], "researcher", note)
        return {
            "research_notes": [*state["research_notes"], note],
            "tool_trace": [
                *state["tool_trace"],
                {"agent": "researcher", "tool": "knowledge_search", "text": search.text},
                {"agent": "researcher", "tool": "memory_recall", "text": recall.text},
            ],
            "trace": [*state["trace"], line],
        }

    def executor(state: WorkflowState) -> dict:
        research = "\n\n".join(state["research_notes"])
        draft = mcp.call_tool("draft_brief", {"goal": state["goal"], "notes": research})
        artifact = llm.execute(state["goal"], research, draft.text).strip() or draft.text
        stored = mcp.call_tool(
            "memory_store",
            {"content": f"Completed brief for: {state['goal'].strip()}", "namespace": "briefs"},
        )
        short_term.add(state["thread_id"], "executor", artifact)
        return {
            "artifacts": [*state["artifacts"], artifact],
            "tool_trace": [
                *state["tool_trace"],
                {"agent": "executor", "tool": "draft_brief", "text": draft.text},
                {"agent": "executor", "tool": "memory_store", "text": stored.text},
            ],
            "trace": [*state["trace"], "executor wrote the brief"],
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("executor", executor)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {"researcher": "researcher", "executor": "executor", "finish": END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("executor", "supervisor")
    return graph.compile()


def _final_answer(state: WorkflowState) -> str:
    if state["artifacts"]:
        return state["artifacts"][-1]
    if state["research_notes"]:
        return (
            state["research_notes"][-1]
            + "\n\nStopped before execution because the step budget was exhausted."
        )
    return "No work completed within the step budget."
