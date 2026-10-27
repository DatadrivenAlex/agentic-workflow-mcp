"""Small local corpus so knowledge search works without the network."""

from __future__ import annotations

import re

CORPUS: list[dict[str, object]] = [
    {
        "title": "LangGraph supervisor",
        "text": (
            "LangGraph compiles a StateGraph of nodes and conditional edges. "
            "A supervisor node chooses the next specialist. The graph stops when the route is finish."
        ),
        "tags": ["langgraph", "supervisor", "routing"],
    },
    {
        "title": "Researcher and executor",
        "text": (
            "The researcher gathers evidence with tools and writes notes. "
            "The executor turns those notes into a deliverable such as a brief."
        ),
        "tags": ["researcher", "executor", "brief"],
    },
    {
        "title": "In-process MCP",
        "text": (
            "Model Context Protocol tools can run in-process. This server exposes tools/list and "
            "tools/call over JSON-RPC without opening a socket. Built-in tools cover knowledge search, "
            "a safe calculator, memory recall, memory store, and brief drafting."
        ),
        "tags": ["mcp", "tools", "json-rpc", "in-process"],
    },
    {
        "title": "Short-term and long-term memory",
        "text": (
            "Short-term memory is a bounded per-thread scratchpad. "
            "Long-term memory is a local SQLite file of facts that later runs can recall."
        ),
        "tags": ["memory", "sqlite", "short-term", "long-term"],
    },
    {
        "title": "Crew-shaped orchestration",
        "text": (
            "A CrewAI-shaped crew has agents with a role, goal, and backstory, plus tasks and a process. "
            "Sequential runs tasks in order. Hierarchical delegates the goal to the LangGraph supervisor."
        ),
        "tags": ["crewai", "crew", "hierarchical", "sequential"],
    },
]

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "with"}


def search_corpus(query: str, limit: int = 3) -> list[dict[str, object]]:
    tokens = [token for token in _TOKEN.findall(query.lower()) if token not in _STOPWORDS and len(token) > 1]
    scored: list[tuple[int, dict[str, object]]] = []
    for document in CORPUS:
        tags = " ".join(str(tag) for tag in document["tags"])  # type: ignore[index]
        haystack = f"{document['title']} {document['text']} {tags}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:limit]]


def format_hits(documents: list[dict[str, object]]) -> str:
    if not documents:
        return "No corpus documents matched the query."
    lines = [f"- {document['title']}: {document['text']}" for document in documents]
    return "\n".join(lines)
