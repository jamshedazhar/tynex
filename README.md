<p align="center">
  <img src="https://raw.githubusercontent.com/jamshedazhar/tynex/main/assets/logo.svg" alt="Tynex logo" width="112" height="112">
</p>

<h1 align="center">Tynex</h1>

<p align="center">
  <strong>A minimal, auditable terminal coding agent harness in Python.</strong><br>
  The same agent loop backs the interactive TUI, headless CLI modes and an embeddable SDK.
</p>

| Package | Responsibility |
| --- | --- |
| `tynex.ai` | Providers, streaming, model catalogs, abort signals |
| `tynex.agent` | Agent loop, tools, permissions, budgets |
| `tynex.coding` | Coding tools, sessions, checkpoints, MCP, prompts, CLI |
| `tynex.tui` | Interactive terminal UI |
| `tynex.sdk` | `Agent` and `@tool` embedding API |

For installation, commands, permissions, sessions, MCP, and configuration, see
the [Tynex User Guide](https://github.com/jamshedazhar/tynex/blob/main/docs/user-guide.md).
Release history is in the [CHANGELOG](https://github.com/jamshedazhar/tynex/blob/main/CHANGELOG.md).

## Features

- **Streaming agent loop** — text, thinking, tool calls, tool results, usage and
  errors are emitted as events.
- **Providers** — Anthropic, OpenAI-compatible (OpenAI, xAI, Groq, DeepSeek,
  Mistral, OpenRouter, Ollama, …), Google Gemini, GitHub Copilot, and an offline
  `faux` provider for tests and demos.
- **Tools** — `read`, `bash`/`powershell`, `edit`, `write`, `submit_plan` and
  `websearch`; `grep`, `find` and `ls` are available to the SDK and to explicit
  tool lists. Searches use the in-process `ripgrep-rs` engine and respect
  `.gitignore`.
- **Permissions** — `plan`, `ask`, `accept-edits` (default) and `bypass`,
  refined by glob rules in `permissions.toml`. Every decision is recorded in the
  session and explained to the model when a call is blocked.
- **Budgets** — opt-in caps on total tokens or cost:
  `--budget cost=2.50` or `--budget tokens=250000`.
- **Sessions** — JSONL session tree with resume, export, labels and automatic compaction.
- **Checkpoints** — `write`/`edit` snapshot the files they change; `/undo`
  restores them.
- **MCP** — stdio and streamable-HTTP servers from `settings.json`, plus the
  Context7 documentation server (on by default).
- **Context and skills** — `AGENTS.md`, `SYSTEM.md`, project and built-in
  `SKILL.md` skills, prompt templates, and extensions.
- **Code review skill** — conditionally runs ruff, sqlfluff, hadolint, tofu and
  tflint, then a verified correctness/security review using repository guidance
  (`AGENTS.md`, `REVIEW.md`) when present, risk tiers and a report.
- **Plan mode** — the agent investigates read-only, submits a plan with
  `submit_plan`, and the turn ends for an explicit approve/reject decision that
  is recorded in the session.
- **Three run modes** — interactive TUI, print, and JSON over stdio.
- **SDK** — typed `@tool` functions and a sync/async `Agent` with events,
  approvals, MCP and budgets.
- **Cross-platform** — Windows, macOS and Linux. File edits preserve line
  endings; the shell resolves to bash/sh when available, otherwise PowerShell.

## Install

Requires Python 3.12 or newer.

```bash
pip install 'tynex[cli]'
tynex login          # API key, OAuth, or a custom provider
tynex
```

The base package is the embedding SDK: `pip install tynex` has two direct
dependencies — `httpx2` (HTTP, keyless web search and page reading) and
`ripgrep-rs` (the engine behind `grep`/`find`/`ls`). `click`, `rich` and
`prompt_toolkit` live in the `cli` extra; a missing extra reports
`pip install 'tynex[cli]'` instead of failing with an import error.

From a checkout:

```bash
uv sync             # installs the CLI and development dependencies
uv run tynex
```

## Quick start

```bash
tynex "explain @src/tynex/cli.py"    # TUI with an initial prompt
tynex -p "list the TODO comments"    # print the final response
tynex --json "summarize main.py" > events.jsonl
tynex -c                             # continue the latest session
tynex -r <id-prefix>                 # resume a specific session
tynex models [provider]              # provider catalogs
tynex sessions                       # sessions for this directory
tynex --plan "add retries to the client"   # plan first, approve, then execute
tynex --help                         # every flag and subcommand

# Run flags may follow the prompt too; `--` ends them.
tynex "fix the failing tests" --permission-mode plan
```

TUI essentials:

- `/` opens the command menu; `@` completes files. `Tab` accepts a completion;
  `Enter` fills it so you can keep typing, then submits on the next `Enter`.
- `/model`, `/thinking`, `/permissions`, `/resume`, `/compact`, `/approve` and
  `/undo` cover the common workflows.
- `!command` runs a shell command and adds its output to the conversation;
  `!!command` runs it without adding the output to the model context.
- `Esc` or `Ctrl+C` interrupts a turn. Text submitted during a turn is queued as
  steering; `/commands` and `!shell` lines return to the editor.
- A tool call is one line until you open it: click the block (or `Ctrl+O`) to
  see its full input and output, and click or `Ctrl+O` again to close it. The
  transcript scrolls on its own — mouse wheel or `PgUp`/`PgDn`.
- `/thinking` levels come from the model's catalog (`deepseek` reports `low`,
  `high`, `max`), and a level the model does not take is clamped to the nearest.
  The choice is stored in the session and in `settings.json`, restored on resume.

## Permissions

Every tool declares a kind — `read`, `web`, `edit`, `shell` or `mcp` — and the
mode decides what runs without asking:

| Mode | Reads & web | File changes | Shell & MCP |
| --- | --- | --- | --- |
| `plan` | allow | deny | deny |
| `ask` | allow | ask | ask |
| `accept-edits` *(default)* | allow | allow | ask |
| `bypass` | allow | allow | allow |

In the TUI an `ask` prints the request above the input line and asks there, in
the same editor: `y` allows once, `a` allows for the session, `n` (or `Esc`)
denies. Headless modes cannot prompt, so `ask` becomes a denial with an
actionable message. The active mode is shown in the startup banner and the
footer, and `/permissions <name>` (or `/permissions mode <name>`) switches it
mid-session.

Rules refine the mode. They live in two files, merged global first and project
last; inside a file the **last matching rule wins**:

```toml
# ~/.tynex/agent/permissions.toml or <project>/.tynex/permissions.toml
mode = "accept-edits"

[[rules]]
tool = "bash"
match = "rm -rf *"
action = "deny"

[[rules]]
tool = "bash"
match = "git *"
action = "allow"
```

`allow` skips the prompt, `deny` blocks the call, and `plan` remains read-only:
rules may tighten it but not widen it. The effective mode comes from
`--permission-mode`, then `permissionMode` in `settings.json`, then `mode` in a
permissions file, then `accept-edits`. `/permissions` shows the mode, rules and
session grants; `/permissions ask` (or `/permissions mode ask`) switches
mid-session.

The workspace guard confines `read`, `write`, `edit`, `grep`, `find` and `ls` to
the directory Tynex was started in: absolute paths, `..` segments and escaping
symlinks are rejected, and `@` completion only lists workspace files. Shell
commands are **not** OS-sandboxed — treat the guard as protection against
mistakes and prompt injection, not as a security boundary for untrusted code.

## Plan mode

`tynex --plan "..."` starts in `plan` mode (`/plan` switches an interactive
session). The agent explores with read-only
tools, then calls `submit_plan`; that tool ends the turn so the plan is reviewed
instead of half-implemented. In the TUI the plan is printed and you choose
`y` (approve and execute) or `n` (keep planning); `/approve` accepts a plan you
deferred. Approving switches to `accept-edits` and records `plan_approved` in
the session; rejecting records `plan_rejected` and leaves plan mode active.

Headless runs print the plan to stdout and keep it in the session (when the
same turn also emitted assistant text, only the plan is printed):

```bash
tynex -p --plan "add retries to the HTTP client"
tynex -c --permission-mode accept-edits "execute the approved plan"
```

Plan mode is enforced by the permission gate, not by the prompt: writes, edits,
shell commands and MCP calls are denied while it is active, and `submit_plan`
is refused outside it. The plan and its decision are session entries, so
resuming an unreviewed plan restores it instead of losing it.

## Budgets

Budgets are opt-in and cap only total tokens or cost:

```bash
tynex -p --budget cost=2.50 "fix the build"
tynex -p --budget tokens=200000 "fix the build"
tynex -p --budget cost=2.50,tokens=200000 "fix the build"
```

Tokens count input + output + cache reads. Without `--budget` nothing is
enforced. When a limit is reached the turn ends normally, a `limit_reached`
event is emitted, the transcript records it, and print mode exits 1.
`/budget` shows the counters for the limits you set.

## MCP servers

Tynex speaks the MCP `2025-06-18` protocol directly, over stdio and streamable
HTTP, with no extra dependency. Context7 is the shipped built-in and is enabled
by default. It is an ordinary `mcpServers` entry in `settings.json` and provides
up-to-date, version-specific library documentation without a key. Set
`"enabled": false` in the global or project settings to turn it off
persistently, or use `--no-mcp` for a single run.

```bash
tynex mcp list                   # built-ins, runtimes, configured servers
tynex mcp add context7 --project # pin it on for this project
tynex mcp remove context7        # remove a project/global override
```

| Built-in | Runtime | Adds |
| --- | --- | --- |
| `context7` | http | Version-specific library and framework documentation |

Custom servers use the same JSON shape:

```bash
tynex mcp add github --command uvx --arg mcp-server-github --env GITHUB_TOKEN=…
tynex mcp add wiki --url https://mcp.deepwiki.com/mcp --header "Authorization=Bearer …"
```

```json
{
  "mcpServers": {
    "local":  { "command": "uvx", "args": ["some-mcp-server"], "env": { "TOKEN": "…" } },
    "remote": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer …" } }
  }
}
```

Settings are deep-merged in three layers: shipped defaults, the global
`~/.tynex/agent/settings.json` (or `$TYNEX_CONFIG_DIR/settings.json`), then the
project's `.tynex/settings.json`. `${workspaceFolder}` (or `{cwd}`) in an
argument expands to the session directory. Servers connect in parallel with an
8-second budget; failures are warnings and do not stop the run. Tools appear as
`mcp__<server>__<tool>` and go through the normal permission policy. `/mcp`
shows live state; `--no-mcp` disables the integration for a run.

## Checkpoints and undo

Before `write` or `edit` changes a file, Tynex copies the previous version into
`~/.tynex/agent/checkpoints/<session>/` (5 MB per file; larger files are recorded
as skipped). `/undo [count]` restores the last `count` write/edit groups
newest-first, deletes files that did not exist before, and adds a note to the
conversation. Disable checkpoints with `"checkpoints": false` in
`settings.json`.

## Providers

| Provider | Credentials |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google | `GEMINI_API_KEY` |
| xAI, Groq, DeepSeek, Mistral, OpenRouter | `XAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY` |
| Ollama | none (local) |
| GitHub Copilot | `tynex login github-copilot`, or `GITHUB_COPILOT_TOKEN` |

Every provider also accepts `settings.apiKeys`. Sign-in stores credentials in
`~/.tynex/agent/auth.json`

```bash
tynex login                   # pick OAuth, API key, or custom provider
tynex login anthropic         # prompt for the API key
tynex login github-copilot    # device flow (asks for the GitHub host)
tynex logout [provider]
```

Tynex ships no hard-coded model names. Catalogs are fetched from each provider's
models API, cached in `~/.tynex/agent/models.json`, refreshed after 24 hours, and
fall back to the cache when the provider is unreachable. Model selection at
startup is: `--model`, the model stored in a resumed session, `model` in
`settings.json`, the newest sign-in in `auth.json`, then a provider environment
variable.

Custom OpenAI-compatible providers can be registered through `tynex login` or
manually:

```json
{
  "providers": {
    "my-llm": {
      "api": "openai-completions",
      "baseUrl": "https://my-llm.example.com/v1",
      "models": [
        {"id": "gpt-oss-120b", "name": "gpt-oss-120b", "contextWindow": 200000, "maxTokens": 16384}
      ]
    }
  }
}
```

**GitHub Copilot.** The device flow works with github.com, GitHub Enterprise
Cloud (`*.ghe.com`) and GitHub Enterprise Server. The catalog is fetched from
your account at login. Optional settings: `providers.github-copilot.enterpriseUrl`
and `providers.github-copilot.clientId`. Environment overrides:
`GITHUB_COPILOT_TOKEN`, `GITHUB_COPILOT_ENTERPRISE_URL`,
`GITHUB_COPILOT_CLIENT_ID`, `GITHUB_COPILOT_API_BASE`.

**Web search.** `websearch` needs no API key. It tries optional Firecrawl
(`FIRECRAWL_API_KEY`) and Brave (`BRAVE_SEARCH_API_KEY`) first, then DuckDuckGo,
Bing RSS and Wikipedia. A `url` argument reads one public page (direct
connections to loopback, private, link-local and multicast addresses are
refused). Proxy environment variables and the OS certificate store apply
through `httpx2`.

## Python SDK

`tynex.sdk` embeds the same agent loop without the CLI or TUI:

```python
from tynex.sdk import Agent, tool

@tool
def search_orders(customer: str, limit: int = 10) -> list[dict]:
    """Find recent orders for a customer.

    Args:
        customer: Customer name or id.
        limit: Maximum number of orders to return.
    """
    return db.search(customer, limit)

agent = Agent(model="anthropic/claude-sonnet-4-5", tools=[search_orders])
result = agent.run("What did Acme order last week?")
print(result.text)
```

- `@tool` derives the JSON schema from type hints (`str`, `int`, `float`, `bool`,
  `list[…]`, `dict[…]`, `Literal[…]`, `Optional[…]`) and reads descriptions from
  the Google-style `Args:` section. Async functions are supported.
- `Agent(...)` accepts `system`, `append_system`, `context`, `extra_guidelines`,
  `thinking`, `temperature`, `max_tokens`, `cwd`, `tools`, `coding_tools`,
  `budget`, `permission_mode`, `permission_rules`, `approval`, `checkpoints`,
  `mcp`/`mcp_servers`, `load_project`, `session` and `on_event`.
- `mcp_servers=[McpServerConfig(...)]` connects at construction and exposes
  `mcp__<server>__<tool>` tools under the normal permission policy.
  `agent.mcp_warnings` reports servers that failed to start.
- SDK defaults: `permission_mode="bypass"`, no checkpoints, no MCP, no project
  discovery (`load_project=True` opts in) and no session file
  (`session="agent.jsonl"` persists).
- `run()` returns `RunResult` with `text`, `messages`, `usage`, `cost`, `steps`,
  `tool_calls`, `session_id`, `stop_reason` and `limit_reason`; `ok` is `False`
  on an error, abort or budget cap.
- `stream()` yields the same events as the CLI; `arun()`/`astream()` are async
  equivalents; `add_tool()` adds tools after construction.

MCP, context and built-in tools combine freely:

```python
from tynex.sdk import Agent, McpServerConfig

agent = Agent(
    model="anthropic/claude-sonnet-4-5",
    mcp_servers=[McpServerConfig(name="time", command="uvx", args=["mcp-server-time"])],
    context=["Team conventions live in docs/style.md."],
    coding_tools=["read", "grep", "ls"],
)
```

An approval callback makes graded modes useful in a service:

```python
from tynex.agent.permissions import PermissionRequest

def approve(request: PermissionRequest) -> str:  # "allow", "session" or "deny"
    return "allow" if request.subject.startswith("git ") else "deny"

agent = Agent(
    model="anthropic/claude-sonnet-4-5",
    coding_tools=["read", "bash"],
    permission_mode="ask",
    approval=approve,
)
```

`examples/sdk_agent.py` combines a custom tool, a budget, streaming and an
approval callback:

```bash
uv run python examples/sdk_agent.py --budget cost=0.50 "tour this repository"
```

## Interactive commands

| Command | Action |
| --- | --- |
| `/help` | show commands |
| `/model [pattern]` | switch model; no argument opens a picker |
| `/thinking [level]` | set the thinking level |
| `/new` | start a new session |
| `/resume`, `/sessions [id]` | browse and switch sessions |
| `/export [path]` | export JSONL branch or static HTML |
| `/compact [instructions]` | compact the conversation |
| `/permissions [mode] <mode>` | show or switch the approval policy |
| `/plan` | switch to plan mode (read-only investigation, then `submit_plan`) |
| `/approve` | accept the pending plan and switch to `accept-edits` |
| `/budget` | show token/cost budget and usage |
| `/undo [count]` | restore files changed by the last write/edit calls |
| `/mcp` | list configured MCP servers and tools |
| `/name [label]` | get or set the session label |
| `/session`, `/stats` | session statistics |
| `/cwd` | print the working directory |
| `/login [provider]`, `/logout [provider]` | manage credentials |
| `/skills`, `/skill:<name> [args]` | list or load a skill |
| `/exit`, `/quit`, `/q` | leave |
| `!command`, `!!command` | run a shell command with or without model-visible output |

## Sessions

Sessions are JSONL files under `~/.tynex/agent/sessions/` (override with
`TYNEX_CONFIG_DIR`). Each line is an entry in a session tree: messages, tool
results, shell executions, model and thinking changes, labels, and compaction
markers. Compaction summarizes older messages and records a `firstKeptEntryId`,
so resume rebuilds the context without resending summarized history.

`/export path.jsonl` writes the active branch; any other path writes a self-contained static HTML transcript.

## Development

```bash
uv sync                  # install/lock dependencies (uv.lock is committed)
uv run pytest            # full suite, fully offline
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m pytest
uv run coverage report      # enforces the 80% coverage floor
```

The suite uses the `faux` provider (`TYNEX_FAUX_SCRIPT` replays a JSON scenario)
and mock HTTP transports, so it needs no API keys or network. CI runs lint,
format check and tests on Python 3.12, 3.13 and 3.14 across Ubuntu, macOS and
Windows (`.github/workflows/ci.yml`).

Infrastructure is wrapped once: `tynex/http.py` owns the single pooled `httpx2`
client (keep-alive, per-request timeouts, SSE streams, and a `set_client()` hook
for tests). Providers, catalogs and web search reuse it.

## Layout

```
src/tynex/
  ai/            types, models, model API clients, abort, providers
  agent/         agent loop, messages/events, permissions, budgets
  agent/skills/  built-in skills shipped in the package
  coding/        tools, web search, sessions, compaction, checkpoints, MCP, prompts
  sdk/           embeddable Agent, @tool, RunResult
  tui/           interactive app, renderer, completion, selector
  cli.py         Click CLI and headless modes
  http.py        pooled HTTP adapter
examples/        runnable SDK example
tests/           offline pytest suite
```

## License

Apache License 2.0. See the `LICENSE` and `NOTICE` files in the distribution.
