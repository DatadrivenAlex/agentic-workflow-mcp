import io
import json

import pytest

from agentic_workflow_mcp.mcp.calculator import evaluate_expression
from agentic_workflow_mcp.mcp.server import InProcessMCPServer, serve_stdio
from agentic_workflow_mcp.memory.store import LongTermMemory


@pytest.fixture
def server(tmp_path):
    return InProcessMCPServer(LongTermMemory(tmp_path / "long_term.sqlite"))


def test_calculator_accepts_arithmetic_and_rejects_calls():
    assert evaluate_expression("(2 + 3) * 4") == "20"
    assert evaluate_expression("2 ** 3") == "8"
    with pytest.raises(ValueError):
        evaluate_expression("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        evaluate_expression("2 ** 20")
    with pytest.raises(ValueError):
        evaluate_expression("1 / 0")


def test_tool_call_shapes(server):
    search = server.call_tool("knowledge_search", {"query": "in-process MCP tools"})
    assert search.is_error is False
    assert "in-process" in search.text.lower()

    math_result = server.call_tool("calculator", {"expression": "7 - 2"})
    assert math_result.text == "5"

    unknown = server.call_tool("shell", {"command": "ls"})
    assert unknown.is_error is True

    invalid = server.call_tool("calculator", {})
    assert invalid.is_error is True


def test_memory_tools_round_trip(server):
    stored = server.call_tool(
        "memory_store",
        {"content": "Completed brief for: in-process MCP tools", "namespace": "briefs"},
    )
    assert stored.is_error is False
    recalled = server.call_tool("memory_recall", {"query": "MCP tools"})
    assert "in-process MCP tools" in recalled.text
    empty = server.call_tool("memory_recall", {"query": "zzzz-unmatched"})
    assert empty.text == "No long-term memories matched."


def test_jsonrpc_initialize_list_call_and_errors(server):
    initialized = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert initialized["result"]["serverInfo"]["name"] == "agentic-workflow-mcp"
    assert initialized["result"]["protocolVersion"] == "2024-11-05"

    listed = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == {"knowledge_search", "calculator", "memory_store", "memory_recall", "draft_brief"}

    called = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "7-2"}},
        }
    )
    assert called["result"]["isError"] is False
    assert called["result"]["content"][0]["text"] == "5"

    failed = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "open('/etc/passwd')"}},
        }
    )
    assert failed["result"]["isError"] is True

    missing = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 5, "method": "tools/nope"})
    assert missing["error"]["code"] == -32601

    ping = server.handle_jsonrpc({"jsonrpc": "2.0", "method": "ping"})
    assert ping is None

    invalid = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 6})
    assert invalid["error"]["code"] == -32600


def test_stdio_serves_one_jsonrpc_message_per_line(server):
    request = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "4*5"}}}
    stdin = io.StringIO(json.dumps(request) + "\n\nnot-json\n")
    stdout = io.StringIO()
    serve_stdio(server, stdin, stdout)
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    ok = json.loads(lines[0])
    assert ok["result"]["content"][0]["text"] == "20"
    bad = json.loads(lines[1])
    assert bad["error"]["code"] == -32700


def test_draft_brief_contains_goal(server):
    result = server.call_tool("draft_brief", {"goal": "Explain MCP", "notes": "Tools run in-process."})
    assert result.text.startswith("# Brief")
    assert "Explain MCP" in result.text
    assert "Tools run in-process." in result.text
