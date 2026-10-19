# Tynex User Guide

Tynex is a terminal coding agent implemented in Python. It can read and edit
files, run commands, search the web, connect to MCP servers, and keep a
resumable transcript of the work. The same agent loop powers the interactive
terminal UI, the print/JSON/RPC modes, and the embedding SDK.

This guide covers everyday CLI use. For the SDK API and package internals, see
the project [README](../README.md).

## Contents

- [Requirements and installation](#requirements-and-installation)
- [First run](#first-run)
- [CLI at a glance](#cli-at-a-glance)
- [Run modes](#run-modes)
- [Built-in tools](#built-in-tools)
- [Interactive terminal UI](#interactive-terminal-ui)
- [Permissions and safety](#permissions-and-safety)
- [Budgets](#budgets)
- [Sessions and undo](#sessions-and-undo)
- [Models and providers](#models-and-providers)
- [MCP servers](#mcp-servers)
- [Project context, skills, and prompts](#project-context-skills-and-prompts)
- [Configuration reference](#configuration-reference)
- [Using Tynex in scripts and CI](#using-tynex-in-scripts-and-ci)
- [Python SDK](#python-sdk)
- [Troubleshooting](#troubleshooting)

## Requirements and installation

Tynex requires **Python 3.12 or newer**.

Install the terminal application:

```bash
pip install 'tynex[cli]'
tynex --version
```

The base package is the embedding SDK:

```bash
pip install tynex
```

The base install has two direct dependencies: `httpx2` for HTTP and
`ripgrep-rs` for workspace search. The CLI extra adds `click`, `rich`, and
`prompt_toolkit`. If those are missing, Tynex tells you to install
`tynex[cli]` rather than failing with a raw import error.

To run from a checkout:

```bash
uv sync
uv run tynex --help
```

`python -m tynex` uses the same entry point as the `tynex` command.

## First run

Start Tynex from the repository or directory you want it to work in:

```bash
cd path/to/project
tynex login
tynex
```

`tynex login` offers three paths:

1. An OAuth device flow (currently GitHub Copilot).
2. An API key for a built-in provider.
3. A custom OpenAI-compatible provider.

If you already have credentials in the environment, you can skip login:

```bash
export ANTHROPIC_API_KEY="..."
tynex "summarize this repository and list the main modules"
```

Useful first checks:

```bash
tynex models                  # list models from every configured provider
tynex models anthropic        # list one provider
tynex --help                  # every run flag and subcommand
tynex login --help
tynex mcp --help
```

Tynex uses the directory where you launch it as the workspace. Start it at the
repository root so project config, context files, and the workspace guard all
line up.

## CLI at a glance

The root command is the run command:

```bash
tynex [OPTIONS] "prompt"
```

There is no separate `run` subcommand. A known first token is treated as a
subcommand; otherwise the whole line is the prompt.

### Run options

| Option | Purpose |
| --- | --- |
| `-p`, `--print` | Run one prompt and print the final response. |
| `--json` | Run one prompt and emit JSON events, one object per line. |
| `--rpc` | Serve the line-delimited JSON RPC protocol over stdin/stdout. |
| `-c`, `-r`, `--resume [ID]` | Resume the latest session, or one matching `ID`. |
| `-m`, `--model MODEL` | Select a model by `provider/model-id`, unique partial id, or glob. |
| `--thinking LEVEL` | Set reasoning level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. |
| `--permission-mode MODE` | `plan`, `ask`, `accept-edits` (default), or `bypass`. |
| `--budget SPEC` | Hard cap: `cost=2.50`, `tokens=250000`, or both comma-separated. |
| `--no-mcp` | Do not start MCP servers for this run. |
| `--no-session` | Do not persist the session to disk. |
| `--append-system-prompt TEXT` | Append text to the system prompt. |
| `-v`, `--version` | Print the version and exit. |
| `-h`, `--help` | Show help and exit. |

`-p` and `--json` can read the prompt from stdin when stdin is not a terminal:

```bash
printf 'list the TODO comments\n' | tynex -p
```

### Subcommands

| Command | Purpose |
| --- | --- |
| `tynex login [provider]` | Sign in with OAuth, store an API key, or add a custom provider. |
| `tynex logout [provider]` | Remove a stored credential from `auth.json`. |
| `tynex mcp list` | Show built-in, configured, and available runtimes. |
| `tynex mcp add NAME` | Enable a built-in server or add a custom MCP server. |
| `tynex mcp remove NAME` | Remove an MCP entry from a settings file. |
| `tynex models [provider]` | List provider model catalogs. |
| `tynex sessions` | List sessions for the current directory. |

A prompt that begins with a subcommand name is dispatched as that command. For
example, `tynex login` opens sign-in; it does not send the word “login” to the
model. Add a leading space to force a prompt if necessary:

```bash
tynex " login"
```

## Run modes

### Interactive mode

```bash
tynex
tynex "start by explaining @src/tynex/cli.py"
```

The interactive mode is the default. It provides streaming output, an editor
with completions, approval panels, session commands, and queued steering while
the model is working.

Interactive mode can offer a sign-in menu when no model or credentials are
available.

### Print mode

```bash
tynex -p "run the tests and fix the first failure"
```

The final assistant response goes to stdout. Tool activity, permission
decisions, compaction notices, and limit messages go to stderr. This keeps
stdout suitable for command substitution and logs.

Exit codes:

- `0` — the turn completed.
- `1` — an unrecoverable error occurred before any response text, or a budget
  limit was reached.
- `130` — the run was interrupted with Ctrl+C.

Print mode cannot open approval panels. If the selected permission mode wants
to ask about a tool call, the call is denied. Add an allow rule, use
`--permission-mode bypass` in a trusted environment, or run interactively.

### JSON event mode

```bash
tynex --json "summarize main.py" > events.jsonl
```

Each line is a complete JSON object with a `type` field. The stream includes
message, tool, permission, compaction, limit, and turn/agent lifecycle events.
It is useful for building a UI or collecting diagnostics:

```bash
tynex --json "review the diff" | jq -c 'select(.type == "message_end")'
```

JSON mode runs one prompt and exits. Only JSON events are written to stdout;
startup warnings such as MCP connection failures may appear on stderr.

### RPC mode

```bash
tynex --rpc
```

RPC mode keeps one agent session alive and exchanges one JSON object per line
over stdin/stdout. The first message announces the session:

```json
{"type":"ready","session":{"id":"<session-id>","file":"<path>","model":"provider/model-id","cwd":"<path>"}}
```

Send a prompt with a request id:

```json
{"id":1,"type":"prompt","message":"list the Python files"}
```

While the prompt runs, the process emits events:

```json
{"type":"event","event":{"type":"message_end","message":{"role":"assistant","content":[]}}}
```

When the command finishes, it sends a response:

```json
{"type":"response","id":1,"command":"prompt","success":true,"data":{"state":{}}}
```

Supported command types:

| Type | Main field | Purpose |
| --- | --- | --- |
| `prompt` | `message` | Run one prompt. |
| `steer` | `message` | Queue a steering message during a run. |
| `follow_up` | `message` | Queue a message for after the current run. |
| `abort` | — | Abort the current run. |
| `get_state` | — | Return session state and active tools. |
| `get_messages` | — | Return the current conversation messages. |
| `set_model` | `model` | Switch models by pattern. |
| `compact` | `instructions` | Compact older messages. |
| `new_session` | — | Start a fresh session. |
| `list_sessions` | — | List sessions for the current directory. |
| `resume_session` | `session` | Switch to a session by id prefix or path. |
| `set_permission_mode` | `mode` | Change the approval policy. |
| `undo` | `count` | Restore the last write/edit groups. |
| `shutdown` | — | End the RPC process. |

RPC mode requires a usable model before it starts, like print and JSON mode.
Prompts are sent over stdin; an initial command-line prompt is not used in RPC
mode.

## Built-in tools

The CLI exposes these tools by default:

| Tool | Kind | What it does |
| --- | --- | --- |
| `read` | read | Read text files with optional offset/limit; images are sent as attachments. |
| `bash` / `powershell` | shell | Run a command in the workspace and return stdout/stderr. |
| `edit` | edit | Apply one or more exact text replacements to a file. |
| `write` | edit | Create or overwrite a file. |
| `websearch` | web | Search the web, or read one public URL. |

The shell tool's output is truncated to the last 2,000 lines or 50 KB,
whichever comes first. If output is truncated, the full text is saved to a
temporary file and its path is included in the result. Shell runs have a
30-minute cap.

The SDK can also enable the read-only `grep`, `find`, and `ls` tools by naming
them in `coding_tools`. They use the in-process `ripgrep-rs` engine and respect
`.gitignore`. The CLI relies on the shell for those operations. MCP servers add
`mcp__<server>__<tool>` tools, which participate in the same permission policy
as built-in tools.

## Interactive terminal UI

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `/` | Open the command menu. |
| `@` | Complete workspace file paths. |
| `Tab` | Accept a completion. |
| `Enter` | With a completion menu open, fill the selected row; press again to submit. |
| `Alt+Enter` | Insert a newline in the prompt. |
| `Esc` | Interrupt the current run. |
| `Ctrl+C` | Interrupt the current run. |
| `Ctrl+D` | Exit Tynex. |
| `y` / `a` / `n` | Approve once / allow this tool for the session / deny. |

Messages typed while a turn is running are queued as steering. A queued
`/command` or `!shell` line is returned to the editor instead of being sent to
the model.

### Commands

| Command | Action |
| --- | --- |
| `/help` | Show commands, shortcuts, aliases, skills, and templates. |
| `/model [pattern]` | Switch model; no argument opens a picker for the current provider. |
| `/thinking [level]` | Set the thinking level; no argument opens a picker. |
| `/new` | Start a new session. |
| `/resume [id]` | Browse or switch sessions. |
| `/tree [entry-id]` | Show the session tree; with an id, branch from that entry. |
| `/fork [entry-id]` | Fork before a user message and put that message back in the editor. |
| `/clone` | Duplicate the current session branch. |
| `/export [path]` | Export the branch as JSONL for `.jsonl`, otherwise as static HTML. |
| `/compact [notes]` | Compact older messages now, optionally with instructions. |
| `/permissions [mode <mode>]` | Show or switch the approval policy. |
| `/budget` | Show token and cost usage against the configured limits. |
| `/undo [count]` | Restore files changed by the last write/edit calls. |
| `/mcp` | List connected MCP servers and their tools. |
| `/name [label]` | Get or set the session label. |
| `/session` | Show session statistics. |
| `/cwd` | Print the working directory. |
| `/login [provider]` | Sign in or store an API key. |
| `/logout [provider]` | Remove a stored credential. |
| `/skills` | List available skills. |
| `/skill:<name> [args]` | Load a skill's instructions into the turn. |
| `/exit`, `/quit`, `/q` | Leave Tynex. |
| `/sessions` | Alias for `/resume`. |
| `/stats` | Alias for `/session`. |

Prompt templates in `.tynex/prompts/` or `prompts/` become commands too. A file
named `fix-tests.md` is invoked as `/fix-tests`.

### Shell escapes

```text
!git status
!!git status
```

A single `!` runs the command and adds its output to the conversation. A
double `!!` runs it without adding the output to the model context. Shell
commands use the same shell tool as the model: `bash`/`sh` when available,
otherwise PowerShell on Windows.

## Permissions and safety

Every tool declares a kind: `read`, `web`, `edit`, `shell`, or `mcp`. The
permission mode decides what runs without asking.

| Mode | Reads and web | File changes | Shell and MCP |
| --- | --- | --- | --- |
| `plan` | allow | deny | deny |
| `ask` | allow | ask | ask |
| `accept-edits` (default) | allow | allow | ask |
| `bypass` | allow | allow | allow |

In the TUI, an `ask` opens an approval panel:

- `y` allows this call once.
- `a` allows that tool for the rest of the session.
- `n` denies it.

Headless modes cannot prompt, so `ask` becomes a denial with a message that
tells the model how to proceed.

### Permission rules

Rules refine the mode. They live in two layers:

```text
~/.tynex/agent/permissions.toml        global
<project>/.tynex/permissions.toml      project, evaluated last
```

Both layers are merged, and the **last matching rule wins**. Project rules
therefore override global rules.

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

[[rules]]
tool = "edit"
match = "src/**"
action = "allow"
```

- `tool` is a glob over the tool name. It defaults to `*`.
- `match` is an optional glob over the call's subject, usually a command,
  path, or URL. String arguments are also considered.
- `action` is `allow`, `ask`, or `deny`.
- `allow` skips the prompt; `deny` blocks the call.

`plan` mode is a hard floor: rules may tighten it but cannot widen it. For
example, an `allow` rule for `bash` does not make shell commands run in plan
mode.

The effective mode is selected in this order:

1. `--permission-mode` on the command line.
2. `permissionMode` in `settings.json`.
3. `mode` in a `permissions.toml` file.
4. `accept-edits`.

Use `/permissions` to inspect the current mode, rules, and session grants, and
`/permissions mode ask` to switch during a session.

### Workspace guard

Tynex confines its file tools to the directory where it was started:

- `read`, `write`, `edit`, `grep`, `find`, and `ls` reject absolute paths,
  `..` segments, and symlinks that escape the workspace.
- `@` completion only lists workspace files.
- The guard is intended to prevent mistakes and reduce prompt-injection
  damage.

The shell is **not** operating-system sandboxed. A shell command can do
anything your user account can do, including accessing files outside the
workspace. MCP servers are also external processes or network services. Run
Tynex with the same trust boundaries you would use for any other tool that can
execute commands.

## Budgets

Budgets are opt-in and cap total tokens or cost for a run:

```bash
tynex -p --budget cost=2.50 "fix the build"
tynex -p --budget tokens=200000 "fix the build"
tynex -p --budget cost=2.50,tokens=200000 "fix the build"
```

Tokens count input, output, and cache reads. Cost is estimated from model
catalog pricing. Without `--budget`, no limit is enforced. The `budget` key in
`settings.json` is intentionally ignored; set limits on the command line or
through the SDK.

When a limit is reached, the turn ends, a `limit_reached` event is emitted, and
the transcript records it. Print mode exits with code 1. Check progress with
`/budget` in the TUI.

## Sessions and undo

### Session storage

Sessions are JSONL files under:

```text
~/.tynex/agent/sessions/
```

Set `TYNEX_CONFIG_DIR` to use a different global config directory. Sessions are
scoped to the directory where Tynex was started, so `tynex sessions` and the
resume picker show only sessions for the current workspace.

Each line in a session is an entry in a tree: messages, tool results, shell
executions, model and thinking changes, labels, and compaction markers.
Compaction summarizes older messages while keeping recent context, so resume
does not resend the summarized history.

### Resuming

```bash
tynex -c                    # continue the latest session for this directory
tynex -r <id-prefix>        # resume a specific session
tynex -r path/to/session.jsonl
tynex sessions              # list available sessions
```

`-r` accepts an id prefix, a file name, or a path. `-c` is the same option with
no id and chooses the most recent session.

Use `--no-session` for a one-off run that should not create a session file:

```bash
tynex -p --no-session "summarize this file"
```

### Fork, clone, export, and tree

- `/fork [entry-id]` creates a new session before an earlier user message and
  puts that message back in the editor.
- `/clone` duplicates the current branch into a new session.
- `/tree` shows the current branch; `/tree <entry-id>` branches from an entry.
- `/export path.jsonl` writes the active branch as JSONL.
- `/export path.html` (or any non-`.jsonl` path) writes a self-contained HTML
  transcript.
- `/name <label>` labels the current session; `/name` shows the label.

### Checkpoints and undo

Before `write` or `edit` changes a file, Tynex snapshots the previous version
under:

```text
~/.tynex/agent/checkpoints/<session>/
```

Files up to 5 MB are copied. Larger files are recorded as skipped and cannot be
restored.

```text
/undo          # restore the most recent write/edit group
/undo 3        # restore the last three groups, newest first
```

Undo deletes files that did not exist before the edit and restores files that
did. It also adds a note to the conversation so the model knows the change was
reverted. Disable checkpoints globally or per project with:

```json
{
  "checkpoints": false
}
```

## Models and providers

Tynex ships no hard-coded model names. It fetches each provider's model catalog,
caches it in `~/.tynex/agent/models.json`, refreshes it after 24 hours, and
falls back to the cache when the provider is unreachable.

### Built-in providers

| Provider | Credentials |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| xAI | `XAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Ollama | none; local endpoint |
| LM Studio | none; local endpoint |
| GitHub Copilot | `tynex login github-copilot`, or `GITHUB_COPILOT_TOKEN` |

Every provider also accepts keys in `settings.apiKeys`. Key resolution order is:

1. `settings.apiKeys[provider]`
2. `~/.tynex/agent/auth.json` (written by `tynex login`)
3. The provider's environment variable

### Signing in and out

```bash
tynex login                   # OAuth, API key, or custom provider menu
tynex login anthropic         # prompt for an Anthropic API key
tynex login github-copilot    # GitHub device flow
tynex logout                  # choose a stored credential
tynex logout anthropic
```

`tynex logout` only removes credentials stored in `auth.json`. It does not
unset environment variables or remove `settings.apiKeys`; it warns when either
still supplies a key.

GitHub Copilot supports github.com, GitHub Enterprise Cloud (`*.ghe.com`), and
GitHub Enterprise Server. Optional environment overrides:

```text
GITHUB_COPILOT_TOKEN
GITHUB_COPILOT_ENTERPRISE_URL
GITHUB_COPILOT_CLIENT_ID
GITHUB_COPILOT_API_BASE
```

### Selecting a model

Use any of these forms with `--model` or `/model`:

```bash
tynex -m anthropic/claude-sonnet-4-5
tynex -m claude-sonnet          # unique partial id
tynex -m 'openai/gpt-*'         # glob
```

Startup selection order is:

1. `--model`
2. The model stored in a resumed session
3. `model` in `settings.json`
4. The newest sign-in in `auth.json`
5. A provider environment variable

If nothing is configured, interactive mode offers sign-in. Print, JSON, and
RPC modes stop with an error.

### Thinking levels

```bash
tynex --thinking high
tynex --thinking off
```

Available levels are `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, and
`max`. A model may support only a subset; Tynex clamps the choice to the
model's available levels. `/thinking` changes are stored in the session and in
the global `settings.json`, so they survive a restart.

### Custom OpenAI-compatible providers

Add a provider in `settings.json`:

```json
{
  "providers": {
    "my-llm": {
      "api": "openai-completions",
      "baseUrl": "https://my-llm.example.com/v1",
      "apiKeyEnv": ["MY_LLM_API_KEY"],
      "models": [
        {
          "id": "gpt-oss-120b",
          "name": "gpt-oss-120b",
          "contextWindow": 200000,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

The interactive `tynex login` custom-provider flow writes this shape for you.
Use `apiKeyEnv` when the key should come from the environment instead of
`auth.json`.

### Web search

The `websearch` tool needs no API key. It tries optional API providers first
when these are set:

```text
FIRECRAWL_API_KEY
BRAVE_SEARCH_API_KEY
```

It then falls back to DuckDuckGo, Bing RSS, and Wikipedia. Passing a URL reads
one public page. Direct connections to loopback, private, link-local, and
multicast addresses are refused. Proxy environment variables and the operating
system certificate store are used through `httpx2`.

## MCP servers

Tynex speaks the MCP `2025-06-18` protocol over stdio and streamable HTTP.
Context7 is the shipped built-in and is enabled by default. It provides
version-specific library documentation without a key.

```bash
tynex mcp list
tynex mcp add context7 --project     # enable/pin it for this project
tynex mcp remove context7 --project  # remove the project override
```

Custom servers:

```bash
tynex mcp add github --command uvx --arg mcp-server-github --env GITHUB_TOKEN=...
tynex mcp add wiki --url https://mcp.deepwiki.com/mcp --header "Authorization=Bearer ..."
```

Or edit the settings JSON directly:

```json
{
  "mcpServers": {
    "local": {
      "command": "uvx",
      "args": ["some-mcp-server"],
      "env": { "TOKEN": "..." }
    },
    "remote": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer ..." }
    }
  }
}
```

Settings are deep-merged in this order:

1. The shipped defaults.
2. The global `~/.tynex/agent/settings.json` (or
   `$TYNEX_CONFIG_DIR/settings.json`).
3. The project `.tynex/settings.json`.

`${workspaceFolder}` and `{cwd}` in a stdio argument expand to the session
directory. Servers connect in parallel with an 8-second budget. A failed server
is reported as a warning and does not stop the run.

Tools appear as `mcp__<server>__<tool>` and go through the normal permission
policy. Use `/mcp` to inspect live connections and `--no-mcp` to disable all
MCP servers for one run:

```bash
tynex --no-mcp "work only with local files"
```

To disable a server persistently, set `"enabled": false` in the relevant
settings layer. A built-in server whose runtime is missing is skipped with a
warning; install the runtime (`npx`, `uvx`, or the configured command) or
disable the server.

## Project context, skills, and prompts

Tynex discovers project files relative to the directory where it was started.
It does not walk parent directories, so launch it at the repository root if you
want root-level context files.

### AGENTS.md and SYSTEM.md

`AGENTS.md` adds repository instructions to the system prompt. The lookup order
is:

1. `<cwd>/.tynex/AGENTS.md`
2. `<cwd>/AGENTS.md`
3. The shipped built-in file, if present

`SYSTEM.md` replaces the default system-prompt preamble. The same lookup order
applies, and a project file wins over the built-in one.

Use these files for conventions the model should always know:

```markdown
# Project rules

- Use `uv run pytest` for tests.
- Keep changes small and update the README when behavior changes.
- Never edit generated files under `dist/`.
```

For a one-off addition, use `--append-system-prompt`:

```bash
tynex --append-system-prompt "Prefer standard-library solutions." "..."
```

The equivalent persistent setting is `appendSystemPrompt`.

### Skills

A skill is a directory containing a `SKILL.md` file:

```text
.tynex/skills/release-check/SKILL.md
skills/release-check/SKILL.md
```

The lookup order is project `.tynex/`, project root, then the shipped built-in
skills. The first definition of a name wins.

```markdown
---
name: release-check
description: Check release readiness and report blockers.
---

# Release check

1. Run the test suite.
2. Check that the version and changelog agree.
3. Report blockers with exact file references.
```

List skills with `/skills` and load one into the current turn:

```text
/skill:release-check
```

The skill body is inserted into the conversation, and the model is told the
skill's directory so it can resolve relative paths.

The built-in `code-review` skill is available without configuration:

```text
/skill:code-review
/skill:code-review origin/main
```

It performs a read-only review, running applicable static-analysis tools when
their file types are in scope, and writes an evidence-backed report
(`report.md` by default). Review guidance is read from `REVIEW.md`,
`AGENTS.md`, and other repository instruction files when present.

### Prompt templates

Markdown files under `.tynex/prompts/` or `prompts/` become TUI commands:

```text
.tynex/prompts/fix-tests.md
```

`/fix-tests` expands the template. Placeholders:

- `{{args}}` expands to all arguments.
- `{{1}}`, `{{2}}`, … expand to positional arguments.
- If the template has no placeholder, provided arguments are appended.

Example:

```markdown
Find and fix the failing test named {{1}}. Run it with uv run pytest and report the result.
```

### Extensions

Extensions are Python modules loaded from `.tynex/extensions/*.py` or
`extensions/*.py` (project files shadow built-ins). A module exposes a
`setup(api)` function:

```python
def setup(api):
    api.add_prompt_section("team-policy", "Never commit generated files.")

    @api.on("tool_call")
    def audit_tool_call(payload):
        # Return {"block": True, "reason": "..."} to stop a call.
        return None

    api.add_command("hello", lambda args: print(f"hello {args}".strip()))
```

The extension API supports:

- `api.add_tool(tool)` — add an `AgentTool`.
- `api.add_command(name, handler, help="")` — add a `/name` command.
- `api.on(event, handler)` — hooks: `tool_call`, `session_start`, `agent_end`.
- `api.add_prompt_section(name, text)` — add a section to the system prompt.

Extensions are loaded in the CLI and TUI. The SDK skips project discovery
unless `load_project=True`.

## Configuration reference

### File locations

The global config directory defaults to `~/.tynex/agent` and can be overridden
with `TYNEX_CONFIG_DIR`.

| Path | Purpose |
| --- | --- |
| `~/.tynex/agent/settings.json` | Global settings. |
| `<project>/.tynex/settings.json` | Project settings, merged over global. |
| `~/.tynex/agent/auth.json` | Credentials written by `tynex login`. |
| `~/.tynex/agent/models.json` | Cached provider model catalogs. |
| `~/.tynex/agent/permissions.toml` | Global permission rules. |
| `<project>/.tynex/permissions.toml` | Project permission rules. |
| `~/.tynex/agent/sessions/` | JSONL session files. |
| `~/.tynex/agent/checkpoints/<session>/` | File snapshots for `/undo`. |
| `~/.tynex/agent/input_history.txt` | Interactive input history. |

Project settings are deep-merged over global settings, which are merged over
the shipped defaults. Objects merge recursively; later scalar values and lists
replace earlier ones.

### Settings keys

| Key | Type | Meaning |
| --- | --- | --- |
| `model` | string | Default model reference. |
| `thinking` | string | Default thinking level. |
| `permissionMode` | string | Default approval mode. |
| `checkpoints` | boolean | Enable file snapshots (default `true`). |
| `appendSystemPrompt` | string | Text appended to the system prompt. |
| `apiKeys` | object | Provider-to-key map, checked before `auth.json`. |
| `providers` | object | Custom OpenAI-compatible provider definitions. |
| `mcpServers` | object | MCP server definitions. |
| `anthropicStealth` | boolean | Use Claude Code tool names for Anthropic requests (default `true`). |

Example global or project settings:

```json
{
  "model": "provider/model-id",
  "thinking": "medium",
  "permissionMode": "accept-edits",
  "checkpoints": true,
  "appendSystemPrompt": "Prefer small, reviewable changes.",
  "apiKeys": {
    "openai": "sk-..."
  }
}
```

Credentials in `settings.json` are plain text. Prefer `tynex login` or
environment variables when you do not want a key stored in the repository
config.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `TYNEX_CONFIG_DIR` | Override the global config directory. |
| `ANTHROPIC_API_KEY` | Anthropic key. |
| `OPENAI_API_KEY` | OpenAI key. |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY` | Google Gemini key. |
| `XAI_API_KEY` | xAI key. |
| `DEEPSEEK_API_KEY` | DeepSeek key. |
| `GROQ_API_KEY` | Groq key. |
| `MISTRAL_API_KEY` | Mistral key. |
| `OPENROUTER_API_KEY` | OpenRouter key. |
| `GITHUB_COPILOT_TOKEN` | GitHub Copilot token. |
| `GITHUB_COPILOT_ENTERPRISE_URL` | GitHub Enterprise host. |
| `GITHUB_COPILOT_CLIENT_ID` | Copilot OAuth client id. |
| `GITHUB_COPILOT_API_BASE` | Copilot API base override. |
| `FIRECRAWL_API_KEY` | Optional web-search provider. |
| `BRAVE_SEARCH_API_KEY` | Optional web-search provider. |

Shell commands started by Tynex also receive these variables:

```text
TYNEX_SESSION_ID
TYNEX_SESSION_FILE
TYNEX_PROVIDER
TYNEX_MODEL
TYNEX_REASONING_LEVEL
```

Proxy variables such as `HTTPS_PROXY` are used by the shared HTTP client, and
the operating system certificate store is respected.

## Using Tynex in scripts and CI

For non-interactive use, choose the mode that matches the output you need:

```bash
# Final text on stdout, tool activity on stderr
tynex -p --no-session "run the tests and summarize failures"

# Machine-readable event stream
tynex --json --no-session "inspect the repository" > events.jsonl

# Read the prompt from stdin
printf 'list the TODOs\n' | tynex -p --no-session
```

In an unattended environment, use `plan` mode for read-only analysis:

```bash
tynex -p --no-session --permission-mode plan "review this change for risks"
```

If the job needs edits or shell commands, provide explicit `permissions.toml`
allow rules rather than relying on interactive approval. `bypass` is only
appropriate in a disposable, trusted sandbox:

```bash
tynex -p --no-session --permission-mode bypass \
  --budget cost=1.00 "apply the requested fix and run the tests"
```

Add `--no-mcp` when a CI job should not connect to external servers. Add
`--no-session` when you do not want transcript files on the runner.

## Python SDK

The SDK embeds the same agent loop without the CLI or TUI:

```python
from tynex.sdk import Agent

agent = Agent(model="provider/model-id", coding_tools=True)
result = agent.run("Summarize this repository.")
print(result.text)
```

Key defaults to remember:

- `permission_mode="bypass"` by default.
- No checkpoints, MCP, project discovery, or session file unless enabled.
- `load_project=True` enables `AGENTS.md`, `SYSTEM.md`, skills, prompts, and
  extensions.
- `session="agent.jsonl"` persists a transcript.
- `stream()` and `astream()` yield the same events as the CLI.

An approval callback is available for service integrations:

```python
from tynex.sdk import Agent

def approve(request):
    return "allow" if request.subject.startswith("git ") else "deny"

agent = Agent(
    model="provider/model-id",
    coding_tools=True,
    permission_mode="ask",
    approval=approve,
)
```

See [README](../README.md) and `examples/sdk_agent.py` for custom tools,
budgets, streaming, and MCP examples.

## Troubleshooting

### “no model available” in print, JSON, or RPC mode

Headless modes require a configured model before they start. Run
`tynex login`, set a provider key, or pass `--model`:

```bash
tynex -m anthropic/claude-sonnet-4-5 -p "hello"
```

Interactive mode can offer the sign-in menu instead of failing.

### Tool calls are denied in print mode

Print, JSON, and RPC modes cannot show approval panels. An `ask` decision
becomes a denial. Options:

- Use `--permission-mode plan` for read-only work.
- Add an allow rule to `permissions.toml`.
- Use `--permission-mode bypass` only in a trusted sandbox.
- Run interactively and approve the call.

### An MCP server does not appear

Start with:

```bash
tynex mcp list
```

Check that the command is on `PATH` (`npx`, `uvx`, or the server's executable),
that the URL and headers are correct, and that the entry is not disabled. The
TUI `/mcp` command shows live state and connection warnings. `--no-mcp` should
be removed if you want servers to start.

### The model catalog is empty

The provider may be unreachable and have no cached catalog yet. Check the API
key, base URL, and network/proxy settings, then run:

```bash
tynex models <provider>
```

Ollama and LM Studio must have their local endpoints running.

### Web search falls back to Wikipedia

Optional search providers are not required. If DuckDuckGo/Bing are blocked by
the network or proxy, the tool falls back to Wikipedia and labels the result.
Set `FIRECRAWL_API_KEY` or `BRAVE_SEARCH_API_KEY` for an API-backed provider,
or configure proxy environment variables.

### A session cannot be resumed

Sessions are scoped to the launch directory. Run `tynex sessions` from the same
directory, or pass a session path explicitly:

```bash
tynex -r /path/to/session.jsonl
```

### Shell commands use a different shell than expected

Tynex prefers `bash` or `sh` on `PATH`, then PowerShell on Windows. Set up the
shell you want on `PATH`, or use the shell that Tynex selects. The tool is not
an OS sandbox; all commands run with your user permissions.

### A prompt is treated as a subcommand

The first token selects a subcommand. Add a leading space, or phrase the prompt
with additional words:

```bash
tynex " login"
tynex "please explain the login flow"
```

Piping the prompt also avoids the dispatch:

```bash
printf 'login\n' | tynex -p
```

## Getting help

- `tynex --help` — run flags and subcommands.
- `tynex <subcommand> --help` — subcommand options.
- `/help` — interactive commands and shortcuts.
- `/permissions`, `/budget`, `/session`, `/mcp`, `/skills` — live session
  state.
- [README](../README.md) — feature overview and SDK examples.

Tynex is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
