"""Command line demo. The default path never calls a model provider."""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from agentic_workflow_mcp.config import Settings, configure_logging
from agentic_workflow_mcp.crew.crew import build_default_crew, build_sequential_crew
from agentic_workflow_mcp.mcp.server import serve_stdio
from agentic_workflow_mcp.runtime import RunReport, WorkflowRuntime

DEMO_GOAL = (
    "Explain how a LangGraph supervisor routes a researcher and an executor "
    "through in-process MCP tools and memory."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentic-workflow",
        description="Run a supervisor, researcher, and executor with in-process MCP tools.",
    )
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL for this process")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Run the built-in offline goal")

    run_parser = subparsers.add_parser("run", help="Run one goal")
    run_parser.add_argument("goal", help="What the crew should produce")
    run_parser.add_argument("--thread", default=None, help="Reuse short-term memory for this id")
    run_parser.add_argument(
        "--process",
        choices=["hierarchical", "sequential"],
        default="hierarchical",
        help="hierarchical uses LangGraph; sequential runs the CrewAI-shaped task list",
    )

    subparsers.add_parser("tools", help="List in-process MCP tools")
    subparsers.add_parser("serve", help="Serve MCP JSON-RPC on stdin/stdout")

    memory_parser = subparsers.add_parser("memory", help="Inspect long-term memory")
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_sub.add_parser("list", help="Show the newest stored facts")
    recall_parser = memory_sub.add_parser("recall", help="Search stored facts")
    recall_parser.add_argument("query")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    try:
        settings = Settings.from_env()
        if args.log_level:
            settings = settings.model_copy(update={"log_level": args.log_level})
    except ValidationError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    try:
        runtime = WorkflowRuntime(settings)
        if args.command == "demo":
            report = build_default_crew(runtime).kickoff({"goal": DEMO_GOAL})
            return _print_crew(runtime, DEMO_GOAL, report.raw, report.tasks)
        if args.command == "run":
            return _run(runtime, args.goal, args.thread, args.process)
        if args.command == "tools":
            for tool in runtime.mcp.list_tools():
                print(f"{tool['name']}: {tool['description']}")
            return 0
        if args.command == "serve":
            serve_stdio(runtime.mcp, sys.stdin, sys.stdout)
            return 0
        if args.command == "memory":
            return _memory(runtime, args.memory_command, getattr(args, "query", ""))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


def _run(runtime: WorkflowRuntime, goal: str, thread_id: str | None, process: str) -> int:
    if not goal.strip():
        raise ValueError("goal is empty")
    if process == "sequential":
        result = build_sequential_crew(runtime).kickoff({"goal": goal})
        print(f"provider: {runtime.llm.name}")
        print("process: sequential")
        print(f"goal: {goal.strip()}")
        print()
        for task in result.tasks:
            print(f"- {task.agent_role}: {task.output}")
        print()
        print("Final answer")
        print("------------")
        print(result.raw)
        return 0
    report = runtime.run(goal, thread_id=thread_id)
    _print_report(report)
    return 0


def _print_crew(runtime: WorkflowRuntime, goal: str, raw: str, tasks) -> int:
    print(f"provider: {runtime.llm.name}")
    print("process: hierarchical")
    print(f"goal: {goal}")
    print()
    for task in tasks:
        print(f"- {task.output}")
    print()
    print("Final answer")
    print("------------")
    print(raw)
    return 0


def _print_report(report: RunReport) -> None:
    print(f"provider: {report.provider}")
    print("process: hierarchical")
    print(f"thread: {report.thread_id}")
    print(f"goal: {report.goal}")
    print()
    for line in report.trace:
        print(f"- {line}")
    print()
    print("Final answer")
    print("------------")
    print(report.final_answer)


def _memory(runtime: WorkflowRuntime, command: str, query: str) -> int:
    if command == "list":
        rows = runtime.long_term.list_recent()
        if not rows:
            print("No long-term memories.")
            return 0
        for row in rows:
            print(f"[{row.id}] ({row.namespace}) {row.content}")
        return 0
    hits = runtime.long_term.search(query)
    if not hits:
        print("No memories matched.")
        return 0
    for row in hits:
        print(f"[{row.id}] ({row.namespace}) {row.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
