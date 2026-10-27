from agentic_workflow_mcp.memory.store import LongTermMemory, ShortTermMemory


def test_short_term_window_persists_and_reloads(tmp_path):
    path = tmp_path / "short_term.json"
    memory = ShortTermMemory(window=2, path=path)
    memory.add("thread-1", "supervisor", "one")
    memory.add("thread-1", "researcher", "two")
    memory.add("thread-1", "executor", "three")

    assert [item["content"] for item in memory.recent("thread-1")] == ["two", "three"]
    reloaded = ShortTermMemory(window=2, path=path)
    assert [item["content"] for item in reloaded.recent("thread-1")] == ["two", "three"]
    assert reloaded.recent("missing") == []


def test_corrupt_short_term_file_starts_empty(tmp_path):
    path = tmp_path / "short_term.json"
    path.write_text("{not json", encoding="utf-8")
    memory = ShortTermMemory(window=4, path=path)
    assert memory.recent("thread-1") == []


def test_long_term_search_ranks_overlap(tmp_path):
    store = LongTermMemory(tmp_path / "long_term.sqlite")
    store.add("Notes about baking bread", namespace="kitchen")
    match = store.add("Completed brief for: in-process MCP tools", namespace="briefs")

    hits = store.search("MCP tools")
    assert hits[0].id == match.id
    assert store.search("zzz-no-such-token") == []
    with_empty = None
    try:
        store.add("   ")
    except ValueError as exc:
        with_empty = str(exc)
    assert with_empty == "memory content is empty"


def test_long_term_round_trip_uses_a_new_connection(tmp_path):
    path = tmp_path / "long_term.sqlite"
    LongTermMemory(path).add("LangGraph supervisor routes the researcher", namespace="briefs")
    reopened = LongTermMemory(path)
    rows = reopened.list_recent()
    assert len(rows) == 1
    assert "supervisor" in rows[0].content
