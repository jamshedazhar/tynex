---
name: code-review
description: Review a change against its stated intent and the repository — conditional static analysis (ruff, sqlfluff, hadolint, tofu/tflint), a blocking security pass for sensitive paths, and a verified LLM correctness/design review. Classify risk and human approvals, drop style nits, mark pre-existing issues, prevent repeated findings, and stop after three rounds. Write an evidence-backed report (report.md by default). Optional inputs — git base ref, report filename/type, previous report, round.
---

# Code review (production workflow)

Find and verify real issues before a human spends time on them. The report is
evidence for a decision, never an approval: `CLEAN` means no findings, not
"merge this." Every finding must trace to an executed command or a quoted
`file:line`.

Review guidance is optional. Look for `REVIEW.md` and `AGENTS.md` first (repo
root or `.tynex/`), then other repository instructions (`CLAUDE.md`,
`.cursorrules`, `.cursor/rules/*`, `.github/copilot-instructions.md`,
`opencode.json`/`.jsonc`). If none exist, infer conventions from the README,
manifests, lint/test/CI configuration, CODEOWNERS, and nearby code/tests; record
`guidance: inferred from repo`. Absence of a guidance file is never a finding.
When guidance conflicts with executable configuration or tests, the executable
source wins; report the stale guidance instead of following it.

Inputs: optional git base ref (whole tree when absent), report filename/type
(`md` or `json`), a previous report, and a round number (default 1).

## Hard rules

1. **Read-only, with one exception.** Never edit project files and never run
   mutating commands (`dbt run`, `tofu apply`, `git commit`, installs). The
   report file itself is the only thing a review writes. Fixes only happen on
   explicit request (step 8).
2. **Conditional tools.** A tool runs only when the file types it covers exist
   in scope (step 2). No dbt project -> sqlfluff is not run, not checked, not
   mentioned as missing.
3. **The review always runs.** Static and security findings never suppress the
   LLM review. Only invalid scope stops the review; an open review with zero
   changed files read is incomplete.
4. **Verified findings only.** Static findings are verified by their tool.
   Every LLM candidate must survive a second-pass code read — ideally with
   fresh context — and exact `file:line` evidence. Drop anything you cannot
   reproduce.
5. **No LLM style findings.** Never produce LLM findings for formatting,
   naming, cleanup, or diagnostic-only issues; static tools cover mechanical
   checks. Behavioral nits are capped at three unless `REVIEW.md` sets another
   cap; nits are never blocking.
6. **No repeats, and converge.** Read the previous report when supplied; do not
   re-report resolved or unchanged findings. Later rounds review only new
   commits. Do not start a fourth round: after three rounds, stop and recommend
   reconsidering the approach.
7. **Security and risk can block.** Sensitive paths require the security pass
   (step 3). Security findings are blocking. Risk is set by the change, not the
   author. This skill reports required human approvals; it never approves,
   merges, or fills a human approval slot.
8. **No silent skips.** An applicable tool or check that is missing or errored
   is listed with its reason.
9. **No invented findings.** Add only what commands or exact code reading
   actually reported.
10. **Every finding is answered.** Blocking findings must be fixed or
    explicitly dismissed with a reason before merge; advisory findings may be
    dismissed. The report marks each finding blocking or advisory; resolution
    and approval happen outside this skill.

## 0. Preflight

Record: repo path, `git rev-parse --short HEAD`, branch, dirty tree
(`git status --porcelain`), and timestamp (ISO 8601).

Find review guidance, in this order. Every step is optional:

1. `REVIEW.md` at the repo root or `.tynex/REVIEW.md`. It may define failure
   modes, paths to skip, a nit cap, and required evidence.
2. `AGENTS.md` at the repo root or `.tynex/AGENTS.md` for project conventions,
   goals, and non-goals.
3. Other instruction files: `CLAUDE.md`, `.cursorrules`, `.cursor/rules/*`,
   `.github/copilot-instructions.md`, and `opencode.json`/`opencode.jsonc`.
4. Repository evidence: `README*`, manifests, lint/test/CI configuration,
   CODEOWNERS, and existing code and tests.

If none of 1–3 exists, use step 4 and record `guidance: inferred from repo`.
Never stop, lower confidence, or add a finding merely because a guidance file
is absent. If guidance conflicts with executable config, scripts, or tests, the
executable source wins; report the stale guidance as a finding. If a change
makes a line in an instruction file out of date, report that too.

Derive the change's intent from the request, PR title/description, commit
messages, or linked issue. State it in the report. If none is available, say
`intent: not stated` and review against the diff and repository conventions;
never invent intent. For a bug fix, identify the root cause and the regression
test that should fail before and pass after; a symptom-only fix is a finding.

Record the verification evidence the change supplies: commands run, test/build
output, screenshots, or a manual test. Missing evidence is a finding for
medium- and high-risk changes.

Read the previous report when supplied and record each finding's identity
(`source`, `rule`, `file`, `line`, message) for repeat detection. `round > 3`
means stop before reviewing: write `round cap reached — step back and
reconsider the approach`. If `round > 1`, review only commits after the previous
report's HEAD plus the working tree; do not re-review unchanged files.

Check tool availability lazily, only after step 2 shows the tool is applicable.
Record `<tool> --version` for every tool that ran.

## 1. Scope and classification

**Input: an optional git base ref.** Take it from the invocation
(`/skill:code-review <ref>`) or the request ("changes since `origin/main`") —
any rev works: branch, tag, `origin/main`, `HEAD~3`, SHA. **No ref -> whole
tree.**

- Base ref given:
  - Validate first: `git rev-parse --verify <ref>^{commit}`. If it does not
    resolve, write the report with `verdict: error — base ref not found` and
    stop. Never guess a nearby ref, never quietly widen to the whole tree.
  - In-scope set — union, then dedupe:
    - `git diff --name-only --diff-filter=ACMR <ref>...HEAD` (commits since
      the merge base)
    - `git diff --name-only --diff-filter=ACMR HEAD` (staged + unstaged)
    - `git ls-files --others --exclude-standard` (untracked; .gitignore
      applies)
  - Deleted paths are excluded (`--diff-filter=ACMR`) — nothing to read.
    Empty set -> `verdict: nothing changed to review` and stop; do not widen
    to the tree.
- No ref -> whole tree: `git ls-files --cached --others --exclude-standard`
  (tracked + untracked; everything `.gitignore` excludes stays out). The
  `find` tool (`pattern: "**"`) enumerates the same way — .gitignore-aware
  through ripgrep — and is the fallback outside a git repo.

- Any round after the first: narrow the in-scope set to the same union with
  the previous report's HEAD in place of the base ref
  (`git diff --name-only --diff-filter=ACMR <prev_head>...HEAD`, working tree,
  and untracked files). Keep the scope computed above when the previous
  report has no HEAD recorded. Unchanged files are never re-read or re-linted.

Read beyond the diff: surrounding functions, callers, tests, interfaces,
migrations, and configuration that the change can affect. Cite those files as
context, but findings still point to a concrete `file:line`.

Never review `.venv`, `node_modules`, `dbt_packages`, `target`, `dist`,
`build`, VCS internals, or this skill's own output file (the resolved report
path, `report.md` by default). Classify every in-scope file into these sets:

| Set | Match |
| --- | --- |
| python | every `*.py` file in scope (source, tests, DAGs, scripts) |
| airflow | `*.py` under a `dags/` (or `dag/`) directory — Airflow DAGs |
| dbt_sql | `*.sql` under a directory containing `dbt_project.yml` |
| dockerfiles | `Dockerfile*`, `Containerfile*` |
| terraform | `*.tf` and `*.tfvars` (under `infra/` or next to `versions.tf`) |

The scope line counts the whole in-scope set — every file of every type
(markdown, YAML, dotfiles included) — not only files that match a set or the
languages a tool ran on; the sets above only drive tool calls.

If every set is empty, the static review is `not applicable`; the LLM review
still runs on the in-scope files. If the whole in-scope set is empty, write
`verdict: nothing to review` with the detection evidence and stop.

## 2. Static analysis

Run a tool only when its set is non-empty; otherwise its status is
`not applicable — no <type> files in scope`. In changed scope the sets contain
only changed files, so e.g. `tofu fmt` runs only when `*.tf` changed and
the Airflow DAG run only when `dags/` files changed.

| Tool | Runs when | Command |
| --- | --- | --- |
| ruff | `python` non-empty | `ruff check <python files> --output-format=json` + `ruff format --check <python files>` |
| sqlfluff | dbt project detected AND `dbt_sql` non-empty | `sqlfluff lint <sql files> --format json --dialect <dialect>` |
| hadolint | `dockerfiles` non-empty | `hadolint --format json <files>` |
| tofu fmt | `terraform` non-empty | `tofu fmt -check -diff <each .tf file>` |
| tofu validate | `terraform` non-empty AND dir has `.terraform/` | `tofu -chdir=<dir> validate -json` |
| tflint | `terraform` non-empty AND `tflint --init` done | `tflint --recursive --format json` in each infra dir |
| ruff — DAGs | `airflow` non-empty | `ruff check <dag files> --select AIR --output-format=json` |

Tool notes:

- Airflow DAGs: `--select AIR` runs only ruff's Airflow rules (standard
  rules already ran on the same files via the Python set) and needs no
  Airflow installation — `AIR301`/`AIR302`/`AIR303` catch Airflow 3
  removals, provider moves and incompatible signatures. Add `--preview` to
  include the preview AIR rules (`AIR003`, `AIR004`, `AIR201`, `AIR202`,
  `AIR304`, `AIR321`) and state the mode in the report.
- sqlfluff dialect: `.sqlfluff` first, else the dbt profile's target type,
  else `ansi` (state the fallback). Use `--templater jinja` when no dbt
  profile is available and say so.
- `tofu fmt` takes file paths — pass the in-scope `.tf` files, not the whole
  directory.
- Never run `tofu init` yourself (writes, may download); without `.terraform/`
  list validate as skipped with "not initialized — run `tofu init
  -backend=false`".
- Give each tool exactly the in-scope paths; for a large full-tree set, the
  directories that cover it exactly (e.g. `src tests`) are acceptable, but in
  changed scope list the changed files.
- Run tools independently: one tool's findings or errors never stop the others.
- Static findings are verified by their tool output. Report them even when the
  LLM review finds nothing, and never drop them because another layer passed.

### Severity mapping

| Tool | Mapping |
| --- | --- |
| ruff check | all -> **medium** (`fix.applicability` marks safe/unsafe fixes) |
| ruff format | formatting differences -> **low** |
| sqlfluff | `PRS`/`TMP` (cannot parse/template) -> **high**; other codes -> **medium** |
| hadolint | error -> **high**, warning -> **medium**, info -> **low**, style -> **info** |
| tofu fmt | any difference -> **high** |
| tofu validate | error diagnostic -> **high**, warning -> **medium** |
| tflint | error -> **high**, warning -> **medium** |
| ruff (AIR) | `AIR301`/`AIR302`/`AIR303` (Airflow 3 removals, provider moves, incompatible signatures) -> **high**; `AIR311`/`AIR312` and other AIR suggestions -> **medium** |

LLM severity: critical (security exploit, data loss, outage), high
(reproducible correctness bug), medium (edge case or weak test), low
(behavioral maintainability issue), info (advisory). Nits are always `info` or
`low` and marked `importance: nit`.

## 3. Security review (blocking when in scope)

Run this when changed paths touch authentication, authorization, sessions or
tokens, secrets/keys, CI workflows, release/install/update code, billing or
payments, permissions, cryptography, network boundaries, migrations or data
deletion, dependency manifests/lockfiles, or container builds.

Use security scanners already configured by the repo (for example semgrep,
gitleaks, trivy, bandit, or pip-audit) without installing anything. Otherwise
perform a focused diff review:

- Secrets committed, logged, or exposed in error paths; cite the path and
  redact the value.
- Injection: shell, SQL, path traversal, SSRF, unsafe deserialization.
- Missing authentication, authorization, tenant isolation, or unsafe defaults.
- CI token scope, untrusted pull-request code, release/install scripts.
- Dependency or supply-chain changes; unexpected lockfile changes.
- Migrations, data deletion, and rollback/reversibility.

Only report concrete `file:line` evidence. Security findings are `critical` or
`high` and blocking; never downgrade them to a nit. If a sensitive path changed
and the security review cannot be completed, the verdict is
`incomplete — security review required`; do not return `CLEAN` and do not
approve.

## 4. LLM review (always)

Run this on every review, including when static analysis or the security pass
found issues. Read all in-scope files (changed files first); for a whole-tree
review over 20 files, sample and state the sampling. Never review zero files
while in-scope files exist.

Ask:

1. **Intent and root cause.** Does the change solve the stated problem? For a
   bug fix, is the cause addressed and is there a regression test? Does it only
   mask the symptom?
2. **Fit.** Does it match the architecture, APIs, data model, configuration,
   permissions, and conventions from available guidance and repository
   evidence? Does it break callers, migrations, or existing behavior? Does it
   make a line in `AGENTS.md`/`REVIEW.md` or another instruction file stale?
3. **Correctness.** Edge cases, error handling, concurrency, resource cleanup,
   security-sensitive flows, and performance.
4. **Verification.** Are the behavior and failure paths tested? Is there
   command/test/build/screenshot evidence? Missing evidence is a finding for
   medium- and high-risk changes.
5. **Operability.** Is the change reversible? Does it need a flag, staged
   release, migration, logging, or rollback path? Could it delete or corrupt
   data?

Do not report formatting, style, naming, cleanup, or diagnostic-only findings;
static tools own those. Do not report a pre-existing issue as a regression.

### Candidate findings

Each candidate is `severity, source: llm, category, file:line, evidence, fix,
origin: introduced|pre-existing|unknown, importance: important|nit`. Categories:
correctness, security, data, performance, tests, operability, architecture.

Behavioral nits are capped at three unless `REVIEW.md` says otherwise. A nit
must still be actionable and tied to behavior or maintainability; drop "could
be nicer" comments.

### Verify every candidate

Before accepting a candidate, re-read the exact code, its callers, and relevant
tests; use a fresh-context verifier when the client supports one. Look for an
existing guard, intentional behavior, or a test that already covers it. Drop
the candidate if it is not reproducible, is intentional, or only repeats a
static-tool finding. Keep only findings with a concrete fix.

For a bug fix, run the narrowest existing non-mutating check that can
demonstrate the behavior when it is safe; otherwise report the evidence that is
missing. Never install dependencies or start services. Never claim a test
passed unless it ran.

Classify `origin` as `pre-existing` when the issue exists at the base ref
(`git show <ref>:path`), `introduced` when the changed lines cause it, and
`unknown` for a whole-tree review with no base. Pre-existing findings are
reported once and do not count against the author; they still block a high-risk
change if the change makes them worse.

Group one root cause across all affected locations: report the class of bug
once, list every instance, and require fixing every instance. This is
especially important for repeated agent fixes.

When a previous report exists, do not repeat its findings. If an old finding is
still open, note it once as `still open` rather than as a new finding.

## 5. Normalize findings

- One finding = severity, source, rule, file, line, col, message, fix, origin,
  importance, blocking, verified.
- `fix` is mandatory and concrete: use the tool-provided fix verbatim when
  present (ruff's `fix`); otherwise give the minimal actionable change from
  the rule's documented guidance (e.g. "Wrap the path in double quotes", "Pin
  the image tag to a digest"). Never restate the message as the fix.
- `blocking` is true for every critical and high finding (every security
  finding falls in one of those tiers); false for medium/low/info and nits.
  `verified` is `tool` for static/security output and `second-pass` for LLM
  findings. A missing required check makes the verdict incomplete rather than
  adding a finding.
- Deduplicate by root cause, not only by message; keep every affected
  `file:line` under one finding.
- Sort: blocking first, then severity (critical -> info), then line; files by
  worst severity, then path.
- If a tool's JSON field names differ by release, read one sample of its
  output and map — never guess.

## 6. Risk tier and approvals

Set the tier from the change, not the author:

| Tier | Examples | Default human approvals |
| --- | --- | --- |
| low | docs, copy, tests only, additive changes behind a flag | 0 when repo policy marks the author trusted; otherwise 1 |
| medium | ordinary product changes; caching, auto-updater, system prompts | 1 |
| high | authentication, billing, CI workflows, release/install/update, permissions, migrations or data deletion, security boundaries | 2; sensitive paths always require a person |

`REVIEW.md` or repository policy can override these defaults. When trust is
unknown, use the stricter tier. An external or untrusted author never gets the
zero-human low-risk path and needs at least one person; high-risk still needs
two. An agent or bot approval never counts as a person's. An approval covers
one specific commit; new commits that the system cannot vouch for need
reapproval. The person who prompted an agent-written change owns the outcome.

Scale review depth with risk: `quick` for low risk, `standard` for medium, and
`deep` for high-risk or security-sensitive changes. State the depth in the
report.

This skill reports `risk` and `approvals_required`; it does not approve,
merge, or fill an approval slot.

## 7. Report contract

**Output path.** Optional `filename` and `type` inputs. Defaults: filename
`report`, type `md` -> `report.md` in the workspace root; a supported
extension on `filename` (`.md`, `.json`) sets the type unless `type` is given
explicitly; `type` alone -> `report.<type>`; a path given directly wins; `-`
prints only, writes nothing. Supported types: `md` (default) and `json` —
anything else: say so and stop without writing. Overwrite the previous
report.

**Type `md`** -> the sections below, in this exact order — keep every section
even when empty; an empty section's body is literally `none`, with no
placeholder table rows. **Type `json`** -> one JSON object with the same data
and no prose: `generated`, `repo`, `git` (sha, branch, dirty), `base_ref` (or
null), `intent`, `guidance`, `risk`, `depth`, `approvals_required`, `round`,
`previous_report`, `scope` (mode, files, counts), `security` (status, reason),
`verdict`, `summary` (counts), `tools[]` (name, version, status, command,
files, findings, reason), `findings[]` (severity, source, rule, file, line,
col, message, fix, origin, importance, blocking, verified), `clean_files[]`
(every path — the markdown cap does not apply), `tool_errors[]`, `llm_review`
(status, reason, files_read, verification, suggestions[]), `reproduce[]`.

1. Title + metadata list
2. Summary — open with the counts (matching the metadata `summary`), then one
   row per tool plus `security`
3. Findings by file — one table per file with findings
4. Clean files — one line (up to ~10 paths; more -> summarize by area
   with counts)
5. Tool errors and skips — one row per entry (columns `tool`, `status`,
   `reason`), including incomplete security
6. LLM review — files read, verification notes, and advisory suggestions table
   (or `no additional findings` + files read); blocking LLM findings are
   already in Findings by file
7. Reproduce — the exact commands in order

Metadata must include intent, guidance, risk and approvals, depth, round and
previous report, and security status. Example:

```markdown
- generated: <ISO timestamp>
- repo: <path> · git <short SHA> (<branch>, clean|dirty)
- intent: <one sentence, or "not stated">
- guidance: REVIEW.md | AGENTS.md | other: <file> | inferred from repo | none
- scope: full tree | changed since <ref> — N files (python a, dbt b, containers c, terraform d, airflow e)
- risk: low|medium|high · depth: quick|standard|deep · approvals required: N
- round: N · previous report: <path|none>
- security: passed | findings | not applicable | incomplete
- verdict: CLEAN | N findings (B blocking; C critical, H high, M medium, L low, I info)
- summary: <counts; "nothing to fix" when clean>
```

When the run stops at scope validation (`error — base ref not found`,
`nothing changed to review`, `nothing to review`, `round cap reached`): the
metadata scope line says `not computed (<reason>)`, sections 2–4 are `none`,
and section 5 carries the reason; Reproduce lists the command that decided it.

Summary statuses: `ran`, `not applicable — <reason>`, `not installed`,
`error`, `incomplete`; spell the reason out (`not applicable — no
Containerfiles in scope`, not the raw set name). `ruff check`, `ruff format`,
`ruff — DAGs`, and `security` get separate rows. Files: `—` for rows that did
not run; a count when the row ran. Findings: `0` for `not applicable` rows; a
count when the row ran; `—` when the count is unknown (not installed, error,
incomplete).

Findings table columns: `# | severity | importance | origin | line | source |
rule | issue | fix`. Show `origin: pre-existing` explicitly; never present it
as the author's regression.

**Never write into the report:** any part of this skill, process narration
("I ran…", "next I…"), raw tool output, chat prose, recommendations not tied
to a file, or secrets. If it is not covered by the template, it does not go
in. Build the tables from the normalized findings — do not retell the review.
The report is written with the write tool in a single call — every section,
Reproduce included — and is never appended to or patched with shell commands
afterwards. Nothing else is written.

````markdown
# Code review report

- generated: <ISO timestamp>
- repo: <path> · git <short SHA> (<branch>, clean|dirty)
- intent: <one sentence>
- guidance: REVIEW.md
- scope: full tree | changed since <ref> — N files (python a, dbt b, containers c, terraform d, airflow e)
- risk: medium · depth: standard · approvals required: 1
- round: 1 · previous report: none
- security: not applicable — no sensitive paths
- verdict: 2 findings (1 blocking; 0 critical, 1 high, 1 medium, 0 low, 0 info)
- summary: 2 findings, 1 blocking, 0 tool errors — fix the blocking finding before merge

## Summary

**2 findings, 1 blocking, 0 tool errors — fix the blocking finding before merge.**

| check | version | status | command | files | findings |
| --- | --- | --- | --- | --- | --- |
| ruff check | 0.17.0 | ran | `ruff check src --output-format=json` | 12 | 1 |
| security | — | not applicable — no sensitive paths | — | — | 0 |

## Findings by file

### `src/app/main.py` — 2 findings (1 blocking, 1 pre-existing)

| # | severity | importance | origin | line | source | rule | issue | fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | high | important | introduced | 42:9 | llm | — | Unchecked error path drops the request | Return the error and cover it with a regression test. |
| 2 | medium | nit | pre-existing | 88:1 | ruff | F401 | Unused import | Remove the import. |

## Clean files

`src/app/util.py` (1 file, 0 findings)

## Tool errors and skips

none

## LLM review

- files read: `src/app/main.py`, `src/app/util.py`
- verification: re-read both files, callers, and tests; one candidate dropped as intentional
- advisory suggestions: none

## Reproduce

```bash
ruff check src --output-format=json
```
````

After writing, state the verdict and counts in chat only — do not echo the
report back.

## 8. Fixes and convergence

Apply only the requested findings, using the smallest change; re-run the
affected tool(s), increment the round, regenerate the report, and re-review
only the new diff. Do not repeat findings that were already reported. After
three rounds, stop and recommend reconsidering the approach instead of running
another review.

If the same class of finding recurs, fix it at the source: add a lint rule,
update `REVIEW.md`/`AGENTS.md`, or update this skill rather than re-reporting
it. If a finding is a false positive, dismiss it with a reason and record the
source fix. Do not let an agent author and an agent reviewer loop on old code.
A materially new change starts a new review at round 1. If a missed finding
causes a revert or incident, add it as a regression case in `REVIEW.md` or the
repository's review tests so the next review catches it.

## CI

Static tool exit codes are the gate (`ruff check` exits 1 on findings;
sqlfluff, hadolint and `tofu fmt -check` likewise). Security findings block as
well. For GitHub annotations use `ruff check --output-format=github`; ruff also
supports `gitlab`, `junit` and `sarif`. The LLM review is evidence for the
report, not an exit code; its blocking findings still must be resolved before
merge. Never let an automated approval replace a required human approval.
