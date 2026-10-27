import pytest

from agentic_workflow_mcp.crew.crew import Agent, Crew, Process, Task, build_default_crew, build_sequential_crew
from agentic_workflow_mcp.runtime import WorkflowRuntime


def test_hierarchical_crew_returns_a_brief(settings):
    runtime = WorkflowRuntime(settings)
    result = build_default_crew(runtime).kickoff(
        {"goal": "Explain how a LangGraph supervisor uses in-process MCP tools and memory."}
    )
    assert result.raw.startswith("# Brief")
    assert any("supervisor -> researcher" in task.output for task in result.tasks)
    assert any("executor wrote the brief" in task.output for task in result.tasks)


def test_sequential_crew_passes_context_and_calls_tools(settings):
    runtime = WorkflowRuntime(settings)
    result = build_sequential_crew(runtime).kickoff({"goal": "in-process MCP tools"})
    assert [task.agent_role for task in result.tasks] == ["Researcher", "Writer"]
    assert "Researcher completed the task. Research in-process MCP tools" in result.tasks[0].output
    assert "Used the previous task output." not in result.tasks[0].output
    assert "Used the previous task output." in result.tasks[1].output
    assert result.raw == result.tasks[-1].output
    assert (settings.memory_dir / "long_term.sqlite").exists()


def test_sequential_crew_reports_a_missing_input(settings):
    runtime = WorkflowRuntime(settings)
    crew = Crew(
        agents=[],
        tasks=[
            Task(
                description="Research {goal}",
                expected_output="notes",
                agent=Agent(role="Researcher", goal="Find facts", backstory="Careful."),
            )
        ],
        process=Process.sequential,
        runtime=runtime,
    )
    with pytest.raises(ValueError, match="task input missing: goal"):
        crew.kickoff({})
