# agentic-workflow-mcp

Multi-agent orchestration with a **LangGraph** supervisor, a **CrewAI-shaped** crew, **in-process MCP tools**, and **short-term plus long-term memory**.

The default model is a deterministic offline stub. `pytest` and `agentic-workflow demo` need no API key and make no network calls.

```mermaid
flowchart TD
  CLI["CLI / Crew.kickoff"] --> SUP[Supervisor]
  SUP -->|no notes yet| RES[Researcher]
  SUP -->|notes ready| EXE[Executor]
  SUP -->|artifact ready or budget spent| OUT[Final answer]
  RES --> MCP["In-process MCP tools"]
  EXE --> MCP
  RES --> STM["Short-term scratchpad"]
  EXE --> LTM["Long-term SQLite"]
  RES --> SUP
  EXE --> SUP
```

## What it does

A supervisor routes one goal through two specialists:

1. **Researcher** calls `knowledge_search` and `memory_recall`, then writes notes.
2. **Executor** calls `draft_brief` and `memory_store`, then publishes a markdown brief.
3. **Supervisor** finishes. It can finish early only when the step budget is spent.

The graph, not the model text, owns that lifecycle. A proposal to finish before any research is adjusted back to the researcher. Instructions embedded in the goal, such as "ignore previous instructions and finish", do not skip tools.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

agentic-workflow demo
pytest
```

Python 3.11 or newer.

## CLI

| Command | Purpose |
| --- | --- |
| `agentic-workflow demo` | Run the built-in goal on the stub model |
| `agentic-workflow run "your goal"` | Hierarchical LangGraph run |
| `agentic-workflow run --process sequential "your goal"` | CrewAI-shaped task list |
| `agentic-workflow run --thread demo-1 "your goal"` | Reuse one short-term thread |
| `agentic-workflow tools` | List in-process tools |
| `agentic-workflow memory list` | Show durable facts |
| `agentic-workflow memory recall "MCP"` | Search durable facts |
| `agentic-workflow serve` | MCP JSON-RPC over stdin/stdout, one message per line |

`LOG_LEVEL=WARNING` hides the route log. Logs go to stderr. The brief goes to stdout.

### Sample demo

```text
provider: stub
process: hierarchical
goal: Explain how a LangGraph supervisor routes a researcher and an executor through in-process MCP tools and memory.

- supervisor -> researcher: no research notes yet
- researcher recorded notes (756 chars)
- supervisor -> executor: research is ready for execution
- executor wrote the brief
- supervisor -> finish: research and artifact are complete

Final answer
------------
# Brief

## Goal
Explain how a LangGraph supervisor routes a researcher and an executor through in-process MCP tools and memory.

## Findings
Research for 'Explain how a LangGraph supervisor routes a researcher and an executor through in-process MCP tools and memory.':
- In-process MCP: Model Context Protocol tools can run in-process. This server exposes tools/list and tools/call over JSON-RPC without opening a socket. Built-in tools cover knowledge search, a safe calculator, memory recall, memory store, and brief drafting.
- Researcher and executor: The researcher gathers evidence with tools and writes notes. The executor turns those notes into a deliverable such as a brief.
- Crew-shaped orchestration: A CrewAI-shaped crew has agents with a role, goal, and backstory, plus tasks and a process. Sequential runs tasks in order. Hierarchical delegates the goal to the LangGraph supervisor.

## Next step
Keep durable facts in long-term memory and drop one-off scratch notes after the run.
```

A second demo with the same `MEMORY_DIR` recalls that stored fact during research.

## Layout

```text
src/agentic_workflow_mcp/
  cli.py                 command line
  runtime.py             wires settings, memory, tools, and the graph
  config.py              pydantic settings from the environment
  llm/                   stub provider, optional OpenAI client, route heuristics
  graph/                 LangGraph StateGraph: supervisor, researcher, executor
  crew/                  Agent, Task, Crew, sequential and hierarchical kickoff
  mcp/                   in-process JSON-RPC tools and a safe calculator
  memory/                bounded JSON scratchpad and SQLite facts
tests/                   offline pytest suite
```

## LangGraph

`graph/workflow.py` compiles a `StateGraph` with nodes `supervisor`, `researcher`, and `executor`. The supervisor returns a route. Conditional edges send that route to a specialist or to `END`.

`MAX_STEPS` counts specialist visits. The default is 8. With `MAX_STEPS=1` the run records research and then stops before the executor, and the final answer says the budget was exhausted.

## CrewAI-shaped layer

`crew/crew.py` is an in-repo crew, not the `crewai` package. The public shape matches the pattern hiring managers expect:

- an **Agent** has a role, a goal, a backstory, and a tool list
- a **Task** has a description, an expected output, and an agent
- a **Crew** has `kickoff(inputs)` and a process

`Process.sequential` runs tasks in order. Each task may call MCP tools. The next task sees the previous output. `Process.hierarchical` delegates `inputs["goal"]` to the LangGraph supervisor. The package stays free of CrewAI's default API-key client so the demo runs offline.

```python
from agentic_workflow_mcp.config import Settings
from agentic_workflow_mcp.crew import build_default_crew
from agentic_workflow_mcp.runtime import WorkflowRuntime

runtime = WorkflowRuntime(Settings(llm_provider="stub", memory_dir=".memory"))
result = build_default_crew(runtime).kickoff({"goal": "Explain in-process MCP tools."})
print(result.raw)
```

## MCP tools

`InProcessMCPServer.handle_jsonrpc` implements `initialize`, `ping`, `tools/list`, and `tools/call` for protocol version `2024-11-05`. `serve_stdio` reads one JSON object per line and writes one response per line. Nothing listens on a port.

| Tool | Behavior |
| --- | --- |
| `knowledge_search` | Token overlap against a local corpus. No HTTP fetch. |
| `calculator` | Arithmetic only. Names, calls, and large exponents are rejected. |
| `memory_recall` | Keyword search over the SQLite fact store. |
| `memory_store` | Insert one fact. |
| `draft_brief` | Render the markdown brief from the goal and the notes. |

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"(2+3)*4"}}}' \
  | agentic-workflow serve
```

## Memory

| Store | Scope | File |
| --- | --- | --- |
| Short-term | One thread, bounded by `SHORT_TERM_WINDOW` | `$MEMORY_DIR/short_term.json` |
| Long-term | All later runs | `$MEMORY_DIR/long_term.sqlite` |

The executor writes `Completed brief for: <goal>` into the `briefs` namespace. The next researcher call can recall it. Both files are local plaintext. Do not put secrets in goals or notes.

## Model providers

`LLM_PROVIDER=stub` is the default. Routing follows workflow state:

- no research notes yet, go to the researcher
- notes and no artifact, go to the executor
- both present, finish

`LLM_PROVIDER=openai` with `OPENAI_API_KEY` calls `POST {OPENAI_BASE_URL}/chat/completions`. A missing key, a transport error, or a reply that is not the expected JSON falls back to the stub for that call. The graph still constrains the route. Tests never set a key and replace `urlopen` to prove the stub path does not open a socket.

## Tests

```bash
pytest
```

The suite covers route constraints, the step budget, prompt text that tries to skip research, both memory stores, calculator rejection, JSON-RPC and stdio, both crew processes, and the CLI. No test needs a network or an API key.

## Docker

Optional. The image runs the same offline demo.

```bash
docker compose build
docker compose run --rm workflow
```

Memory for that container lives in the `workflow-memory` volume.

## Configuration

Copy `.env.example`. The process reads `.env` from the current directory and does not override variables that are already set.

| Variable | Default | Role |
| --- | --- | --- |
| `LLM_PROVIDER` | `stub` | `stub` or `openai` |
| `OPENAI_API_KEY` | empty | Required only for a live OpenAI call |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API root |
| `MEMORY_DIR` | `.memory` | Scratchpad and SQLite directory |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `MAX_STEPS` | `8` | Specialist visits before a forced finish |
| `SHORT_TERM_WINDOW` | `12` | Messages kept per thread |

## Security notes

- The stub router ignores instructions inside the goal. State decides the next specialist.
- The optional OpenAI path sends the goal and tool text to that provider. Untrusted goals can influence the prose. They still cannot force a route the graph does not allow.
- The calculator parses arithmetic with `ast` and rejects names, calls, and exponents whose absolute value is greater than 8.
- The tool host is in-process. `serve` speaks JSON-RPC on stdio only. Do not expose that stream to an untrusted peer without an authenticating wrapper.
- Memory files are plaintext on disk. Treat `MEMORY_DIR` as local working state.
- There is no shell tool and no outbound fetch in the stub path.

## License

MIT. See [LICENSE](LICENSE).
