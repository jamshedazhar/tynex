# AGENTS.md

Tynex (`tynex`) is a Python 3.12+ terminal coding agent harness. One agent loop
backs the TUI, headless print/JSON/RPC modes, and the SDK.

## Commands

- `uv sync` — install CLI and development dependencies.
- `uv run pytest -q` — full offline suite. One test:
  `uv run pytest tests/test_mcp.py::test_name -q`.
- `uv run ruff check .` and `uv run ruff format --check .` — CI gates. Write
  formatting with `uv run ruff format .`.
- `uv run coverage run -m pytest` — local coverage only; CI does not run it.
  `uv run coverage report` enforces the 80% floor.
- `uv run tynex ...` from the checkout; the installed entrypoint is `tynex`, and
  `python -m tynex` uses the same `tynex.cli:main`.
- `uv build` — wheel and sdist. CI runs `uv sync --locked`, then ruff and
  pytest on Python 3.12/3.13/3.14 across Ubuntu, macOS, and Windows. There is
  no typecheck or pre-commit hook.

## Architecture

- `tynex.ai`: provider adapters, streaming, model registry.
- `tynex.agent`: agent loop, permissions, budgets.
- `tynex.coding`: tools, sessions, checkpoints, MCP, prompts, CLI.
- `tynex.tui`: interactive terminal UI. `tynex.sdk`: embedded `Agent` API.
- `AgentSession` (`tynex/coding/agent_session.py`) wires the model, tools, MCP,
  permissions, prompts, and sessions. Start there for behavior changes.
- `src/tynex/cli.py` is intentionally a single file: group routing, run flags,
  subcommands, and headless modes. Do not split it without being asked.
- All network calls go through `tynex/http.py` (pooled `httpx2`). Tests install a
  mock client with `set_client()`; the autouse fixture resets it.
- Model catalogs are fetched and cached; there are no hardcoded model IDs. Bump
  `CACHE_VERSION` in `tynex/coding/model_catalog.py` if the cached schema changes.

## CLI invariants

- Root `tynex` is the run command. Subcommands are `login`, `logout`, `mcp`,
  `models`, and `sessions`; there is no `run` subcommand.
- `TynexGroup` dispatches a known first token as a subcommand, otherwise the
  whole line is the prompt. A prompt beginning with a subcommand name will
  dispatch that command.
- Permissions are selected only with `--permission-mode`
  (`plan|ask|accept-edits|bypass`); per-tool rules live in `permissions.toml`.
  Do not reintroduce `--tools`, `--allow`, `--ask`, or `--deny`.
- Budgets are opt-in via `--budget cost=...` or `--budget tokens=...` (or both).
  Only tokens and cost are capped; `budget` in settings.json is intentionally
  ignored.
- Keep `tynex --help` showing every run flag and subcommand; tests assert this.
- README is the user-facing contract. Update it when CLI, settings, or provider
  behavior changes, and keep branding standalone with no upstream/port
  references.

## Tests

- The suite is offline. Never call real providers or MCP servers.
- `tests/conftest.py` autouse fixtures isolate `TYNEX_CONFIG_DIR`, strip provider
  keys, and make `load_settings()` see shipped network MCP servers disabled.
  `builtin_catalog()` still reads the real shipped settings; keep that split so
  catalog tests stay truthful and runtime tests stay offline.
- Use `faux_model` with `faux_scenario`/`text_scenario`; `TYNEX_FAUX_SCRIPT`
  replays scenarios for subprocesses.
- MCP tests use a local fake stdio script or local HTTP server; see
  `tests/test_mcp.py`. Do not add live Context7 calls to the default suite.
- CLI subprocess tests must use `cli_env()` in `tests/test_integration.py`; it
  writes the model cache and explicitly disables the shipped Context7 server.
- A skipped file-symlink test on Windows is expected.

## Settings and data

- Settings deep-merge shipped `src/tynex/agent/settings.json`, then global
  `~/.tynex/agent/settings.json` (`TYNEX_CONFIG_DIR` overrides the directory), then
  project `.tynex/settings.json`. Context7 ships enabled by default.
- `permissions.toml` is loaded global first, then project; the last matching rule
  wins.
- Keep web search keyless and dependency-free: it uses `tynex/http.py` providers
  with fallback. Do not add a required search API key or SDK.

## Packaging

- Base install is exactly `httpx2` and `ripgrep-rs`. `click`, `rich`, and
  `prompt_toolkit` belong only in the `cli` extra.
- Version is single-sourced from `src/tynex/__init__.py` via
  `dynamic = ["version"]`; do not add a static version to pyproject.
- Run `uv lock` after pyproject changes. `[tool.uv] exclude-newer` pins
  resolution; bump it before resolving newer packages.

## Serialization gotchas

- Preserve Gemini thought signatures (`TextContent.signature`,
  `ToolCall.signature`) when serializing sessions or replaying provider
  requests. Gemini 3 rejects replayed function calls without them.
- `tynex/http.py` is the only HTTP adapter; providers, catalogs, and web search
  must reuse it rather than creating their own clients.
