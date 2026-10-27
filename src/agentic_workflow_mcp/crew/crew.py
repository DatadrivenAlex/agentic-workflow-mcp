"""A small CrewAI-shaped API on top of the same runtime.

Sequential process runs each task in order and feeds the previous output forward.
Hierarchical process delegates the goal to the LangGraph supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agentic_workflow_mcp.runtime import WorkflowRuntime


class Process(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    tools: list[str] | None = None

    def __post_init__(self) -> None:
        self.tools = list(self.tools or [])


@dataclass
class Task:
    description: str
    expected_output: str
    agent: Agent


@dataclass
class TaskOutput:
    description: str
    agent_role: str
    output: str


@dataclass
class CrewResult:
    raw: str
    tasks: list[TaskOutput]


@dataclass
class Crew:
    agents: list[Agent]
    tasks: list[Task]
    process: Process
    runtime: WorkflowRuntime

    def __post_init__(self) -> None:
        self.process = Process(self.process)

    def kickoff(self, inputs: dict[str, str] | None = None) -> CrewResult:
        values = inputs or {}
        if self.process is Process.hierarchical:
            goal = str(values.get("goal", "")).strip()
            if not goal and self.tasks:
                goal = self.tasks[0].description
            report = self.runtime.run(goal)
            return CrewResult(
                raw=report.final_answer,
                tasks=[
                    TaskOutput(description=line, agent_role="workflow", output=line) for line in report.trace
                ],
            )
        outputs: list[TaskOutput] = []
        context = ""
        for task in self.tasks:
            try:
                description = task.description.format(**values)
            except KeyError as exc:
                missing = exc.args[0]
                raise ValueError(f"task input missing: {missing}") from exc
            tool_sections = []
            for tool_name in task.agent.tools or []:
                result = self.runtime.mcp.call_tool(tool_name, _tool_arguments(tool_name, description, context, values))
                tool_sections.append(f"[{tool_name}]\n{result.text}")
            user = description
            if context:
                user += f"\n\nContext from earlier tasks:\n{context}"
            if tool_sections:
                user += "\n\nTool results:\n" + "\n\n".join(tool_sections)
            system = f"Role: {task.agent.role}\nGoal: {task.agent.goal}\n{task.agent.backstory}"
            output = self.runtime.llm.complete(system, user)
            outputs.append(TaskOutput(description=description, agent_role=task.agent.role, output=output))
            context = output
        raw = outputs[-1].output if outputs else ""
        return CrewResult(raw=raw, tasks=outputs)


def build_sequential_crew(runtime: WorkflowRuntime) -> Crew:
    researcher = Agent(
        role="Researcher",
        goal="Collect evidence with MCP tools",
        backstory="You search the local corpus and recall long-term memory.",
        tools=["knowledge_search", "memory_recall"],
    )
    writer = Agent(
        role="Writer",
        goal="Turn the research into a short completion",
        backstory="You write from the notes you are given.",
    )
    return Crew(
        agents=[researcher, writer],
        tasks=[
            Task(
                description="Research {goal}",
                expected_output="Research notes",
                agent=researcher,
            ),
            Task(
                description="Write a short brief for {goal}",
                expected_output="A short brief",
                agent=writer,
            ),
        ],
        process=Process.sequential,
        runtime=runtime,
    )


def build_default_crew(runtime: WorkflowRuntime) -> Crew:
    supervisor = Agent(
        role="Supervisor",
        goal="Route work until a brief exists",
        backstory="You coordinate specialists and you do not invent tool results.",
    )
    researcher = Agent(
        role="Researcher",
        goal="Collect evidence with MCP tools",
        backstory="You search the local corpus and recall long-term memory.",
        tools=["knowledge_search", "memory_recall"],
    )
    executor = Agent(
        role="Executor",
        goal="Publish a brief and store a durable fact",
        backstory="You turn research notes into a short markdown brief.",
        tools=["draft_brief", "memory_store"],
    )
    return Crew(
        agents=[supervisor, researcher, executor],
        tasks=[
            Task(
                description="Produce a brief for: {goal}",
                expected_output="A markdown brief",
                agent=supervisor,
            )
        ],
        process=Process.hierarchical,
        runtime=runtime,
    )


def _tool_arguments(tool_name: str, description: str, context: str, inputs: dict[str, str]) -> dict:
    goal = str(inputs.get("goal") or description)
    if tool_name == "calculator":
        return {"expression": str(inputs.get("expression") or description)}
    if tool_name == "memory_store":
        return {"content": context or description, "namespace": "crew"}
    if tool_name == "draft_brief":
        return {"goal": goal, "notes": context or description}
    return {"query": goal}
