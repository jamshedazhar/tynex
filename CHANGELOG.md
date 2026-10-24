# Changelog

## [0.2.0] - 2026-10-24

### Added
- Plan mode: run `tynex --plan "..."`. Use `/plan`. It checks with tools calls `submit_plan` and ends the turn for an explicit approve or reject. Use `/approve` to accept a plan you had deferred. Approvals and rejections are stored as session entries so an unreviewed plan will survive if the session is resumed.
- Collapsible tool blocks in the TUI: a call appears as one line until you click it or press `Ctrl+O`. The transcript scrolls automatically using the mouse wheel or `PgUp`/`PgDn`.
- Thinking levels are taken from the model catalog. If a model does not support a level it is clamped to the nearest available level. OpenAI‑compatible models that advertise levels receive a matching `reasoning_effort` value.
- flags may follow the prompt, e.g. `Tynex "fix the failing tests" --permission-mode plan`. A double dash `--` ends flag parsing.

### Changed
- The TUI is now built on a prompt that contains the transcript, status, editor and toolbar in a single prompt_toolkit application. Approvals are answered directly in the editor than in a separate panel.
- `/permissions <name>` now switches modes without requiring the `mode` keyword. The active mode is shown in the startup banner and in the footer.
- `submit_plan` now joins the default tool set, which includes `read` the shell, `edit` `write` and `websearch`.

### Removed
- The run mode (`--rpc`). Run modes are now limited to the TUI, print and JSON.
- The `/fork` `/clone` and `/tree` commands.

### Fixed
- Token budgets now count both input and output when a provider reports no `total_tokens` and cache writes are included.
- `/undo` now skips entries that resolve outside the workspace and displays a warning instead of restoring them.
- `auth.json` is created with owner‑only permissions (`0600`).
- Permission rules now match the tools permission subject (falling back to `path` `command` `query` `url` `glob` `pattern` or `name`) of matching every string argument.
- A tool that ends the turn (`) now does so even when it is called alongside other tools.

## [0.1.0] - 2026-10-19

### Added
- public release: interactive TUI, print and JSON modes, the embeddable SDK, provider adapters, for Anthropic, OpenAI‑compatible endpoints Google Gemini and GitHub Copilot (plus an offline `faux` provider) filesystem and shell tools, permissions, budgets, sessions, checkpoints and MCP.