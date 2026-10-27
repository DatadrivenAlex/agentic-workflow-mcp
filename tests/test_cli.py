from agentic_workflow_mcp.cli import main


def test_help_returns_zero(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "demo" in output
    assert "memory" in output


def test_demo_is_offline_and_prints_the_brief(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["demo"]) == 0
    output = capsys.readouterr().out
    assert "provider: stub" in output
    assert "supervisor -> researcher" in output
    assert "Final answer" in output
    assert "# Brief" in output


def test_tools_and_memory_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["tools"]) == 0
    assert "calculator" in capsys.readouterr().out

    assert main(["demo"]) == 0
    capsys.readouterr()
    assert main(["memory", "list"]) == 0
    listed = capsys.readouterr().out
    assert "Completed brief" in listed

    assert main(["memory", "recall", "MCP"]) == 0
    assert "Completed brief" in capsys.readouterr().out


def test_run_rejects_a_blank_goal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    assert main(["run", "   "]) == 2
    assert "goal is empty" in capsys.readouterr().err


def test_sequential_process_from_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    assert main(["run", "--process", "sequential", "in-process MCP tools"]) == 0
    output = capsys.readouterr().out
    assert "process: sequential" in output
    assert "Researcher completed the task." in output


def test_invalid_provider_is_a_config_error(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    assert main(["tools"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err
