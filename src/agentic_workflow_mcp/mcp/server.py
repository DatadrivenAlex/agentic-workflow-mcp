"""JSON-RPC tool server that stays inside the Python process."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TextIO

from pydantic import BaseModel, Field, ValidationError

from agentic_workflow_mcp import __version__
from agentic_workflow_mcp.mcp.calculator import evaluate_expression
from agentic_workflow_mcp.mcp.corpus import format_hits, search_corpus
from agentic_workflow_mcp.memory.store import LongTermMemory

_PROTOCOL_VERSION = "2024-11-05"


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=1)


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1)


class MemoryStoreArgs(BaseModel):
    content: str = Field(min_length=1)
    namespace: str = "default"


class MemoryRecallArgs(BaseModel):
    query: str = Field(min_length=1)


class DraftBriefArgs(BaseModel):
    goal: str = Field(min_length=1)
    notes: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    text: str
    is_error: bool = False


class InProcessMCPServer:
    """MCP-shaped tool host. `handle_jsonrpc` never opens a socket."""

    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory
        self._handlers = {
            "knowledge_search": self._knowledge_search,
            "calculator": self._calculator,
            "memory_store": self._memory_store,
            "memory_recall": self._memory_recall,
            "draft_brief": self._draft_brief,
        }
        self._models: dict[str, type[BaseModel]] = {
            "knowledge_search": KnowledgeSearchArgs,
            "calculator": CalculatorArgs,
            "memory_store": MemoryStoreArgs,
            "memory_recall": MemoryRecallArgs,
            "draft_brief": DraftBriefArgs,
        }
        self._descriptions = {
            "knowledge_search": "Search the local knowledge corpus. No network access.",
            "calculator": "Evaluate a safe arithmetic expression.",
            "memory_store": "Save a fact in long-term SQLite memory.",
            "memory_recall": "Find long-term memories that share words with the query.",
            "draft_brief": "Render a markdown brief from a goal and research notes.",
        }

    def list_tools(self) -> list[dict]:
        tools = []
        for name, description in self._descriptions.items():
            tools.append(
                {
                    "name": name,
                    "description": description,
                    "inputSchema": self._models[name].model_json_schema(),
                }
            )
        return tools

    def call_tool(self, name: str, arguments: dict | None = None) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(name=name, text=f"Unknown tool: {name}", is_error=True)
        model = self._models[name]
        try:
            parsed = model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult(name=name, text=f"Invalid arguments: {exc.errors()[0]['msg']}", is_error=True)
        try:
            return handler(parsed)
        except ValueError as exc:
            return ToolResult(name=name, text=str(exc), is_error=True)

    def handle_jsonrpc(self, message: dict) -> dict | None:
        if not isinstance(message, dict) or not isinstance(message.get("method"), str):
            return _error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
        method = message["method"]
        request_id = message.get("id")
        notification = "id" not in message
        try:
            result = self._dispatch(method, message.get("params") or {})
        except _RpcError as exc:
            payload = _error(request_id, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            payload = _error(request_id, -32603, str(exc))
        else:
            payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        if notification and "error" not in payload:
            return None
        return payload

    def _dispatch(self, method: str, params: object) -> dict:
        if not isinstance(params, dict):
            raise _RpcError(-32602, "Invalid params")
        if method == "initialize":
            return {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "agentic-workflow-mcp", "version": __version__},
            }
        if method == "ping":
            return {"ok": True}
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                raise _RpcError(-32602, "Invalid params")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise _RpcError(-32602, "Invalid params")
            result = self.call_tool(name, arguments)
            return {
                "content": [{"type": "text", "text": result.text}],
                "isError": result.is_error,
            }
        raise _RpcError(-32601, "Method not found")

    def _knowledge_search(self, args: KnowledgeSearchArgs) -> ToolResult:
        return ToolResult(name="knowledge_search", text=format_hits(search_corpus(args.query)))

    def _calculator(self, args: CalculatorArgs) -> ToolResult:
        return ToolResult(name="calculator", text=evaluate_expression(args.expression))

    def _memory_store(self, args: MemoryStoreArgs) -> ToolResult:
        record = self.memory.add(args.content, namespace=args.namespace)
        text = f"Stored memory {record.id} in namespace '{record.namespace}'."
        return ToolResult(name="memory_store", text=text)

    def _memory_recall(self, args: MemoryRecallArgs) -> ToolResult:
        records = self.memory.search(args.query)
        if not records:
            return ToolResult(name="memory_recall", text="No long-term memories matched.")
        lines = [f"- ({record.namespace}) {record.content}" for record in records]
        return ToolResult(name="memory_recall", text="\n".join(lines))

    def _draft_brief(self, args: DraftBriefArgs) -> ToolResult:
        findings = args.notes.strip() or "No research notes were recorded."
        brief = (
            "# Brief\n\n"
            f"## Goal\n{args.goal.strip()}\n\n"
            f"## Findings\n{findings}\n\n"
            "## Next step\n"
            "Keep durable facts in long-term memory and drop one-off scratch notes after the run.\n"
        )
        return ToolResult(name="draft_brief", text=brief)


def serve_stdio(server: InProcessMCPServer, stdin: TextIO, stdout: TextIO) -> None:
    """Read one JSON-RPC message per line and write one response per line.

    Notifications produce no output. This is the stdio shape MCP clients use,
    without binding a socket.
    """

    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, -32700, "Parse error")
        else:
            response = server.handle_jsonrpc(message if isinstance(message, dict) else {})
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
